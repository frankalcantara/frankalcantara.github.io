/**
 * Fallback simples para Kyrion Forms
 * Versão básica que funciona sem dependências externas
 */

// Estado global simplificado
window.KyrionForms = {
    currentRoute: 'home',
    forms: [],
    responses: [],
    settings: {
        theme: 'light', // Forçar tema claro
        autoSave: true
    }
};

// Classe de aplicação simplificada
class SimpleKyrionApp {
    constructor() {
        console.log('Construindo SimpleKyrionApp...');
        
        // Sistema de múltiplos formulários
        this.formsRegistry = new Map(); // Map de formRegistry por ID
        this.currentFormId = null;
        
        // Sistema de clipboard para questões
        this.questionsClipboard = []; // Array de questões copiadas
        
        // Compatibilidade com código existente
        this.questionsRegistry = null; // Será referência ao questionsRegistry do formulário ativo
        
        this.init();
        
        // Expor métodos globalmente para debug
        window.editTitle = () => this.editTitle();
        window.editDescription = () => this.editDescription();
        console.log('Metodos de edicao expostos globalmente');
    }
    
    init() {
        console.log('Kyrion Forms - Modo Simplificado');
        
        // Configurar Marked.js
        this.setupMarkdown();
        
        // Aplicar tema
        this.updateTheme();
        
        // Configurar navegação básica
        this.setupNavigation();
        
        // Configurar interface
        this.setupUI();
        
        // Carregar página inicial
        this.loadHome();
        
        // Renderizar cards iniciais (vazio inicialmente)
        this.renderFormCards();
        
        console.log('Aplicacao carregada com sucesso');
    }
    
    /**
     * Configura Marked.js para renderização de markdown
     */
    setupMarkdown() {
        // Aguardar carregamento do Marked.js se necessário
        const checkMarked = () => {
            if (typeof marked !== 'undefined') {
                // Criar renderer customizado para tabelas
                const renderer = new marked.Renderer();
                
                // Customizar renderização de tabelas
                renderer.table = function(header, body) {
                    return `<div class="markdown-table-container">
                        <table class="markdown-table">
                            <thead>${header}</thead>
                            <tbody>${body}</tbody>
                        </table>
                    </div>`;
                };
                
                marked.setOptions({
                    async: false,
                    breaks: false,
                    extensions: null,
                    gfm: true,
                    hooks: null,
                    pedantic: false,
                    silent: false,
                    tokenizer: null,
                    walkTokens: null,
                    renderer: renderer,
                    headerIds: false,
                    mangle: false,
                    sanitize: false
                });
                console.log('Marked.js configurado com renderer customizado para tabelas');
                return true;
            }
            return false;
        };
        
        if (!checkMarked()) {
            console.log('Aguardando carregamento do Marked.js...');
            // Tentar novamente após um breve delay
            setTimeout(() => {
                if (!checkMarked()) {
                    console.warn('Marked.js nao encontrado apos delay');
                }
            }, 100);
        }
    }
    
    /**
     * Renderiza markdown para HTML mantendo estilos existentes
     * @param {string} markdownText - Texto em markdown
     * @returns {string} HTML renderizado
     */
    renderMarkdown(markdownText) {
        if (!markdownText || !markdownText.trim()) {
            console.log('renderMarkdown: texto vazio');
            return '';
        }
        
        if (typeof marked === 'undefined') {
            console.warn('renderMarkdown: Marked.js nao disponivel, usando fallback');
            // Fallback: retornar texto simples se Marked.js não estiver disponível
            return markdownText.replace(/\n/g, '<br>');
        }
        
        try {
            // Garantir que o texto está em UTF-8 correto
            const cleanText = markdownText.normalize('NFC');
            const rendered = marked.parse(cleanText);
            console.log('renderMarkdown: sucesso', { input: cleanText, output: rendered });
            return rendered;
        } catch (error) {
            console.error('Erro ao renderizar markdown:', error);
            return markdownText.replace(/\n/g, '<br>');
        }
    }
    
    setupNavigation() {
        // Menu mobile toggle
        const menuButton = document.getElementById('menu-button');
        const sidebar = document.getElementById('sidebar');
        
        if (menuButton && sidebar) {
            menuButton.addEventListener('click', () => {
                // Mobile: toggle sidebar
                if (window.innerWidth <= 768) {
                    sidebar.classList.toggle('open');
                } else {
                    // Desktop: toggle collapsed
                    sidebar.classList.toggle('collapsed');
                }
            });
        }
        
        // Navegação da sidebar
        const navButtons = document.querySelectorAll('[data-route]');
        navButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const route = button.getAttribute('data-route');
                
                // Verificar se é um botão expansível (menu principal "Formulários")
                if (button.classList.contains('nav-button-expandable')) {
                    const navItem = button.closest('.nav-item-expandable');
                    if (navItem) {
                        // Toggle expanded state
                        navItem.classList.toggle('expanded');
                        // Continue para navegar também (não fazer return aqui)
                    }
                }
                
                // Remover classe ativa de todos os botões
                navButtons.forEach(btn => btn.classList.remove('active'));
                
                // Adicionar classe ativa ao botão clicado
                button.classList.add('active');
                
                // Se for um submenu button, também expandir o menu pai
                if (button.classList.contains('submenu-button')) {
                    const parentNavItem = button.closest('.nav-item-expandable');
                    if (parentNavItem) {
                        parentNavItem.classList.add('expanded');
                        // Também marcar o botão pai como ativo se for rota de formulários
                        const parentButton = parentNavItem.querySelector('.nav-button-expandable');
                        if (parentButton && route.startsWith('forms')) {
                            parentButton.classList.add('active');
                        }
                    }
                }
                
                // Navegar para a rota
                this.navigate(route);
                
                // Fechar sidebar em mobile
                if (window.innerWidth <= 768) {
                    sidebar.classList.remove('open');
                }
            });
        });
        
        // FAB
        const fab = document.getElementById('main-fab');
        if (fab) {
            fab.addEventListener('click', () => {
                // Se estamos na página new-form, criar questão. Senão, navegar para new-form
                if (window.location.hash === '#new-form' || document.querySelector('.questions-container')) {
                    this.createMultipleChoiceQuestion('fab');
                } else {
                    this.navigate('new-form');
                }
            });
        }
    }
    
    setupUI() {
        // Marcar botão ativo inicial
        const homeButton = document.querySelector('[data-route="home"]');
        if (homeButton) {
            homeButton.classList.add('active');
        }
        
        // Fechar sidebar ao clicar fora (mobile)
        document.addEventListener('click', (e) => {
            const sidebar = document.getElementById('sidebar');
            const menuButton = document.getElementById('menu-button');
            
            if (window.innerWidth <= 768 && 
                sidebar && 
                !sidebar.contains(e.target) && 
                !menuButton.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }
    
    navigate(route = 'home') {
        console.log(`Navegando para: ${route}`);
        
        // Destruir registry se saindo da página de criação de formulário
        const currentRoute = window.KyrionForms.currentRoute;
        if (currentRoute === 'new-form' && route !== 'new-form') {
            this.destroyFormRegistry();
        }
        
        window.KyrionForms.currentRoute = route;
        
        switch (route) {
            case 'home':
                this.loadHome();
                break;
            case 'forms':
                this.loadFormsOverview();
                break;
            case 'forms-created':
                this.loadForms();
                break;
            case 'forms-answered':
                this.loadFormsAnswered();
                break;
            case 'forms-grading':
                this.loadFormsGrading();
                break;
            case 'new-form':
                this.loadNewForm();
                break;
            case 'responses':
                this.loadResponses();
                break;
            default:
                this.loadHome();
        }
    }
    
    loadHome() {
        const content = document.getElementById('content');
        if (!content) return;
        
        content.innerHTML = `
            <div class="home-container">
                <div class="page-header">
                    <h1>Bem-vindo ao Kyrion Forms</h1>
                    <p>Crie formulários inteligentes com suporte a LaTeX e código</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="material-icons">description</i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-number">${window.KyrionForms.forms.length}</div>
                            <div class="stat-label">Formulários</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="material-icons">analytics</i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-number">${window.KyrionForms.responses.length}</div>
                            <div class="stat-label">Respostas</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="material-icons">trending_up</i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-number">0</div>
                            <div class="stat-label">Média por Form</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="material-icons">schedule</i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-number">0</div>
                            <div class="stat-label">Esta Semana</div>
                        </div>
                    </div>
                </div>
                
                <div class="quick-actions">
                    <h2>Ações Rápidas</h2>
                    <div class="actions-grid">
                        <div class="action-card" onclick="window.simpleApp.navigate('new-form')">
                            <div class="action-icon">
                                <i class="material-icons">add_circle</i>
                            </div>
                            <h3>Novo Formulário</h3>
                            <p>Crie um formulário com questões múltipla escolha e texto livre</p>
                        </div>
                        
                        <div class="action-card" onclick="window.simpleApp.navigate('forms-created')">
                            <div class="action-icon">
                                <i class="material-icons">folder</i>
                            </div>
                            <h3>Meus Formulários</h3>
                            <p>Visualize e gerencie todos os seus formulários</p>
                        </div>
                        
                        <div class="action-card" onclick="window.simpleApp.navigate('responses')">
                            <div class="action-icon">
                                <i class="material-icons">assignment</i>
                            </div>
                            <h3>Ver Respostas</h3>
                            <p>Analise as respostas recebidas dos formulários</p>
                        </div>
                        
                        <div class="action-card" onclick="alert('Funcionalidade em desenvolvimento')">
                            <div class="action-icon">
                                <i class="material-icons">upload</i>
                            </div>
                            <h3>Importar</h3>
                            <p>Importe formulários de outros sistemas</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Página inicial carregada com sucesso
        console.log('Pagina inicial carregada');
    }
    
    loadFormsOverview() {
        const content = document.getElementById('content');
        if (!content) return;
        
        // Verificar se há formulários criados
        const hasFiles = this.formsRegistry && this.formsRegistry.size > 0;
        
        content.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1>Formulários</h1>
                    <p>Gerencie todos os seus formulários em um só lugar</p>
                    <button class="primary-button" onclick="window.simpleApp.navigate('new-form')">
                        <i class="material-icons">add</i>
                        Novo Formulário
                    </button>
                </div>
                
                <!-- Seção Criados -->
                <div class="forms-section">
                    <div class="section-header">
                        <h2>Formulários Criados</h2>
                        <button class="secondary-button" onclick="window.simpleApp.navigate('forms-created')">
                            Ver todos
                        </button>
                    </div>
                    
                    ${hasFiles ? `
                        <div class="forms-grid" id="forms-grid-created">
                            ${this.generateFormsCards()}
                        </div>
                    ` : `
                        <div class="empty-state-mini">
                            <div class="empty-icon">
                                <i class="material-icons">description</i>
                            </div>
                            <h3>Nenhum formulário criado ainda</h3>
                            <p>Comece criando seu primeiro formulário</p>
                        </div>
                    `}
                </div>
                
                <!-- Seção Respondidos -->
                <div class="forms-section">
                    <div class="section-header">
                        <h2>Formulários Respondidos</h2>
                        <button class="secondary-button" onclick="window.simpleApp.navigate('forms-answered')">
                            Ver todos
                        </button>
                    </div>
                    
                    <div class="empty-state-mini">
                        <div class="empty-icon">
                            <i class="material-icons">assignment_turned_in</i>
                        </div>
                        <h3>Nenhum formulário respondido ainda</h3>
                        <p>Os formulários respondidos aparecerão aqui</p>
                    </div>
                </div>
                
                <!-- Seção Em Correção -->
                <div class="forms-section">
                    <div class="section-header">
                        <h2>Formulários em Correção</h2>
                        <button class="secondary-button" onclick="window.simpleApp.navigate('forms-grading')">
                            Ver todos
                        </button>
                    </div>
                    
                    <div class="empty-state-mini">
                        <div class="empty-icon">
                            <i class="material-icons">grading</i>
                        </div>
                        <h3>Nenhum formulário em correção</h3>
                        <p>Formulários aguardando correção aparecerão aqui</p>
                    </div>
                </div>
            </div>
        `;
        
        // Se há formulários criados, configurar event listeners
        if (hasFiles) {
            this.setupFormsPageListeners();
        }
    }
    
    loadForms() {
        const content = document.getElementById('content');
        if (!content) return;
        
        // Verificar se há formulários criados
        const hasFiles = this.formsRegistry && this.formsRegistry.size > 0;
        
        if (!hasFiles) {
            // Mostrar estado vazio
            content.innerHTML = `
                <div class="page-container">
                    <div class="page-header">
                        <h1>Formulários Criados</h1>
                        <p>Gerencie todos os seus formulários criados</p>
                    </div>
                    
                    <div class="empty-state">
                        <div class="empty-icon">
                            <i class="material-icons">description</i>
                        </div>
                        <h3>Nenhum formulário criado ainda</h3>
                        <p>Comece criando seu primeiro formulário</p>
                        <button class="primary-button" onclick="window.simpleApp.navigate('new-form')">
                            <i class="material-icons">add</i>
                            Criar Formulário
                        </button>
                    </div>
                </div>
            `;
        } else {
            // Mostrar formulários como cards
            content.innerHTML = `
                <div class="page-container">
                    <div class="page-header">
                        <h1>Formulários Criados</h1>
                        <p>Gerencie todos os seus formulários criados</p>
                        <button class="primary-button" onclick="window.simpleApp.navigate('new-form')">
                            <i class="material-icons">add</i>
                            Novo Formulário
                        </button>
                    </div>
                    
                    <div class="forms-grid" id="forms-grid">
                        ${this.generateFormsCards()}
                    </div>
                </div>
            `;
            
            // Configurar event listeners para os cards
            this.setupFormsPageListeners();
        }
    }
    
    /**
     * Gera HTML dos cards de formulários para a página "Meus Formulários"
     * @returns {string} HTML dos cards
     */
    generateFormsCards() {
        if (!this.formsRegistry || this.formsRegistry.size === 0) {
            return '';
        }
        
        return Array.from(this.formsRegistry.values())
            .sort((a, b) => b.modified - a.modified) // Mais recente primeiro
            .map(form => this.generateFormPageCard(form))
            .join('');
    }
    
    /**
     * Gera HTML de um card de formulário para a página "Meus Formulários"
     * @param {Object} formRegistry - Dados do formulário
     * @returns {string} HTML do card
     */
    generateFormPageCard(formRegistry) {
        const questionsCount = formRegistry.metadata.totalQuestions || 0;
        const responsesCount = Math.floor(Math.random() * 50); // Mockup de respostas
        const timeText = formRegistry.metadata.estimatedTime > 0 ? `${formRegistry.metadata.estimatedTime}min` : '0min';
        const modifiedDate = new Date(formRegistry.modified).toLocaleDateString('pt-BR');
        
        return `
            <div class="form-page-card" data-form-id="${formRegistry.id}">
                <div class="form-page-card-header">
                    <div class="form-page-card-icon">
                        <i class="material-icons">${formRegistry.icon}</i>
                    </div>
                    <div class="form-page-card-actions">
                        <button class="card-action-btn" data-action="edit" title="Editar">
                            <i class="material-icons">edit</i>
                        </button>
                        <button class="card-action-btn" data-action="copy" title="Duplicar">
                            <i class="material-icons">content_copy</i>
                        </button>
                        <button class="card-action-btn" data-action="delete" title="Excluir">
                            <i class="material-icons">delete</i>
                        </button>
                    </div>
                </div>
                
                <div class="form-page-card-content">
                    <h3 class="form-page-card-title">${formRegistry.name}</h3>
                    <p class="form-page-card-description">${formRegistry.description}</p>
                </div>
                
                <div class="form-page-card-stats">
                    <div class="stat-item">
                        <i class="material-icons">quiz</i>
                        <span>${questionsCount} questões</span>
                    </div>
                    <div class="stat-item">
                        <i class="material-icons">people</i>
                        <span>${responsesCount} respostas</span>
                    </div>
                    <div class="stat-item">
                        <i class="material-icons">schedule</i>
                        <span>${timeText}</span>
                    </div>
                </div>
                
                <div class="form-page-card-footer">
                    <small class="form-modified-date">Modificado em ${modifiedDate}</small>
                    <div class="form-status ${formRegistry.status}">
                        ${formRegistry.status === 'published' ? 'Publicado' : 'Rascunho'}
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Configura event listeners para os cards da página "Meus Formulários"
     */
    setupFormsPageListeners() {
        const cards = document.querySelectorAll('.form-page-card');
        
        cards.forEach(card => {
            const formId = card.dataset.formId;
            
            // Click no card para editar formulário
            card.addEventListener('click', (e) => {
                if (e.target.closest('.card-action-btn')) return; // Ignorar clicks nos botões de ação
                
                this.loadExistingForm(formId);
            });
            
            // Botões de ação
            const editBtn = card.querySelector('[data-action="edit"]');
            const copyBtn = card.querySelector('[data-action="copy"]');
            const deleteBtn = card.querySelector('[data-action="delete"]');
            
            if (editBtn) {
                editBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.loadExistingForm(formId);
                });
            }
            
            if (copyBtn) {
                copyBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.duplicateForm(formId);
                });
            }
            
            if (deleteBtn) {
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.deleteFormFromPage(formId);
                });
            }
        });
    }
    
    /**
     * Duplica um formulário
     * @param {string} formId - ID do formulário a ser duplicado
     */
    duplicateForm(formId) {
        const originalForm = this.formsRegistry.get(formId);
        if (!originalForm) return;
        
        // Criar cópia do formulário
        const duplicatedForm = {
            ...originalForm,
            id: `form_${Date.now()}`,
            name: `${originalForm.name} (Cópia)`,
            created: new Date(),
            modified: new Date(),
            metadata: {
                ...originalForm.metadata,
                lastAccessed: new Date()
            },
            questionsRegistry: originalForm.questionsRegistry.map(question => ({
                ...question,
                domElement: null // Limpar referência DOM
            }))
        };
        
        // Adicionar ao registry
        this.formsRegistry.set(duplicatedForm.id, duplicatedForm);
        
        // Recarregar página
        this.loadForms();
        
        // Atualizar sidebar
        this.renderFormCards();
        
        console.log(`Formulario duplicado: ${duplicatedForm.id}`);
    }
    
    /**
     * Exclui um formulário da página "Meus Formulários"
     * @param {string} formId - ID do formulário a ser excluído
     */
    deleteFormFromPage(formId) {
        const formRegistry = this.formsRegistry.get(formId);
        if (!formRegistry) return;
        
        const confirmDelete = confirm(`Tem certeza que deseja excluir o formulário "${formRegistry.name}"?`);
        if (!confirmDelete) return;
        
        // Remover do registry
        this.formsRegistry.delete(formId);
        
        // Se for o formulário ativo, desativar
        if (this.currentFormId === formId) {
            this.currentFormId = null;
            this.questionsRegistry = null;
        }
        
        // Recarregar página
        this.loadForms();
        
        // Atualizar sidebar
        this.renderFormCards();
        
        console.log(`Formulario excluido: ${formId}`);
    }
    
    loadNewForm() {
        console.log('Criando novo formulario...');
        
        // SEMPRE criar um novo formulário quando este método é chamado
        // Este método é chamado pelo botão "Novo Formulário" da sidebar
        this.initializeFormRegistry();
        
        const content = document.getElementById('content');
        if (!content) {
            console.error('Elemento content nao encontrado');
            return;
        }
        
        console.log('Elemento content encontrado');
        content.innerHTML = `
            <div class="page-container">
                <div class="form-header-editable">
                    <div class="editable-title-group">
                        <h1 id="form-title-display" class="editable-title">
                            Novo Formulário
                        </h1>
                        <button type="button" class="edit-btn edit-title-btn" title="Editar título">
                            <i class="material-icons">edit</i>
                        </button>
                        <input 
                            type="text" 
                            id="form-title-input" 
                            class="title-input hidden"
                            value="Novo Formulário"
                        >
                    </div>
                    
                    <div class="editable-description-group">
                        <p id="form-description-display" class="editable-description">
                            Crie um formulário com questões personalizadas
                        </p>
                        <button type="button" class="edit-btn edit-description-btn" title="Editar descrição">
                            <i class="material-icons">edit</i>
                        </button>
                        <textarea 
                            id="form-description-input" 
                            class="description-input hidden"
                            rows="2"
                        >Crie um formulário com questões personalizadas</textarea>
                    </div>
                    
                    <div class="form-navigation-tabs">
                        <div class="nav-tabs-left">
                            <button type="button" class="nav-tab-icon active" data-tab="questions" title="Perguntas">
                                <i class="material-icons">quiz</i>
                            </button>
                            <button type="button" class="nav-tab-icon" data-tab="responses" title="Respostas">
                                <i class="material-icons">analytics</i>
                            </button>
                            <button type="button" class="nav-tab-icon" data-tab="preview" title="Visualização">
                                <i class="material-icons">visibility</i>
                            </button>
                        </div>
                        <div class="nav-tabs-right">
                            <button type="button" class="nav-tab-icon" data-tab="settings" title="Configurações">
                                <i class="material-icons">settings</i>
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Canvas de Edição -->
                <div class="form-canvas">
                    <!-- Aba de Perguntas -->
                    <div id="questions-tab" class="tab-content active">
                        <div class="questions-container">
                            <!-- Container vazio - questões criadas dinamicamente via JavaScript -->
                        </div>
                        
                        <!-- FAB - Floating Action Bar -->
                        <div class="floating-action-bar">
                            <button class="fab-btn" data-action="add-question" title="Adicionar pergunta">
                                <i class="material-icons">add</i>
                            </button>
                            <button class="fab-btn" data-action="import-questions" title="Importar perguntas">
                                <i class="material-icons">file_download</i>
                            </button>
                            <button class="fab-btn" data-action="add-image" title="Adicionar imagem">
                                <i class="material-icons">image</i>
                            </button>
                            <button class="fab-btn" data-action="add-video" title="Adicionar vídeo">
                                <i class="material-icons">videocam</i>
                            </button>
                            <button class="fab-btn" data-action="add-section" title="Adicionar seção">
                                <i class="material-icons">view_agenda</i>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Aba de Respostas -->
                    <div id="responses-tab" class="tab-content">
                        <div class="responses-placeholder">
                            <i class="material-icons">analytics</i>
                            <h3>Nenhuma resposta ainda</h3>
                            <p>Assim que as pessoas responderem, você verá as informações aqui.</p>
                        </div>
                    </div>
                    
                    <!-- Aba de Visualização -->
                    <div id="preview-tab" class="tab-content">
                        <div class="preview-container">
                            <div class="preview-content">
                                <div class="form-preview">
                                    <div class="preview-form-header">
                                        <h2 id="preview-title">Título do Formulário</h2>
                                        <p id="preview-description">Descrição do formulário aparecerá aqui</p>
                                    </div>
                                    <div id="preview-questions">
                                        <!-- Questões serão renderizadas aqui -->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Aba de Configurações -->
                    <div id="settings-tab" class="tab-content">
                        <div class="questions-container">
                            <div class="question-block selected" data-question-id="settings">
                                <div class="question-header">
                                    <h2 style="flex: 1; margin: 0; color: var(--md-sys-color-on-primary); font-size: 1.25rem; font-weight: 500;">Configurações do Formulário</h2>
                                    <select class="question-type" style="min-width: 200px;">
                                        <option value="general" selected>Configurações Gerais</option>
                                        <option value="access">Controle de Acesso</option>
                                        <option value="display">Exibição</option>
                                        <option value="advanced">Avançado</option>
                                    </select>
                                </div>
                                
                                <div class="question-options" style="padding: var(--spacing-lg);">
                                    <div class="settings-group">
                                        <h4 style="color: var(--md-sys-color-on-surface); margin: 0 0 var(--spacing-md) 0; font-size: 1rem; font-weight: 500;">Coleta de Dados</h4>
                                        <label class="setting-item">
                                            <input type="checkbox" class="setting-checkbox">
                                            <span>Coletar endereços de e-mail</span>
                                        </label>
                                        <label class="setting-item">
                                            <input type="checkbox" class="setting-checkbox">
                                            <span>Requerer login para responder</span>
                                        </label>
                                    </div>
                                    
                                    <div class="settings-group" style="margin-top: var(--spacing-lg);">
                                        <h4 style="color: var(--md-sys-color-on-surface); margin: 0 0 var(--spacing-md) 0; font-size: 1rem; font-weight: 500;">Controle de Respostas</h4>
                                        <label class="setting-item">
                                            <input type="checkbox" class="setting-checkbox">
                                            <span>Limitar a uma resposta por pessoa</span>
                                        </label>
                                        <label class="setting-item">
                                            <input type="checkbox" class="setting-checkbox">
                                            <span>Permitir edição após envio</span>
                                        </label>
                                    </div>
                                    
                                    <div class="settings-group" style="margin-top: var(--spacing-lg);">
                                        <h4 style="color: var(--md-sys-color-on-surface); margin: 0 0 var(--spacing-md) 0; font-size: 1rem; font-weight: 500;">Exibição</h4>
                                        <label class="setting-item">
                                            <input type="checkbox" class="setting-checkbox" checked>
                                            <span>Mostrar barra de progresso</span>
                                        </label>
                                        <label class="setting-item">
                                            <input type="checkbox" class="setting-checkbox">
                                            <span>Embaralhar ordem das perguntas</span>
                                        </label>
                                        <label class="setting-item">
                                            <input type="checkbox" class="setting-checkbox">
                                            <span>Mostrar numeração das perguntas</span>
                                        </label>
                                    </div>
                                    
                                    <div class="settings-group" style="margin-top: var(--spacing-lg);">
                                        <h4 style="color: var(--md-sys-color-on-surface); margin: 0 0 var(--spacing-md) 0; font-size: 1rem; font-weight: 500;">Tempo e Submissão</h4>
                                        <div class="setting-item" style="align-items: flex-start; flex-direction: column; gap: var(--spacing-sm);">
                                            <label style="display: flex; align-items: center; gap: var(--spacing-sm);">
                                                <input type="checkbox" class="setting-checkbox" id="time-limit-enabled">
                                                <span>Definir limite de tempo</span>
                                            </label>
                                            <div style="display: flex; align-items: center; gap: var(--spacing-sm); margin-left: var(--spacing-lg);">
                                                <input type="number" 
                                                       style="width: 80px; padding: var(--spacing-xs); border: 1px solid var(--md-sys-color-outline); border-radius: var(--radius-sm);" 
                                                       placeholder="60" 
                                                       min="1" 
                                                       max="300">
                                                <span style="color: var(--md-sys-color-on-surface-variant); font-size: 0.875rem;">minutos</span>
                                            </div>
                                        </div>
                                        <label class="setting-item">
                                            <input type="checkbox" class="setting-checkbox">
                                            <span>Confirmar antes de enviar</span>
                                        </label>
                                    </div>
                                </div>
                                
                                <div class="question-footer">
                                    <div class="question-actions">
                                        <button class="action-btn" title="Resetar configurações">
                                            <i class="material-icons">refresh</i>
                                        </button>
                                        <button class="action-btn" title="Exportar configurações">
                                            <i class="material-icons">download</i>
                                        </button>
                                        <button class="action-btn more-options" title="Mais opções">
                                            <i class="material-icons">more_vert</i>
                                        </button>
                                    </div>
                                    <div class="question-settings">
                                        <span style="color: var(--md-sys-color-on-surface-variant); font-size: 0.875rem;">
                                            <i class="material-icons" style="font-size: 16px; vertical-align: middle;">info</i>
                                            Configurações aplicadas automaticamente
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        console.log('Configurando form creator...');
        this.setupFormCreator();
    }
    
    /**
     * Carrega um formulário existente para edição
     * @param {string} formId - ID do formulário a ser carregado
     */
    loadExistingForm(formId) {
        console.log(`Carregando formulario existente: ${formId}`);
        
        // Ativar o formulário especificado
        const activated = this.activateForm(formId);
        if (!activated) {
            console.error(`Nao foi possivel ativar formulario: ${formId}`);
            return;
        }
        
        // Obter dados do formulário atual
        const currentForm = this.getCurrentFormRegistry();
        if (!currentForm) {
            console.error('Formulario ativo nao encontrado');
            return;
        }
        
        // Carregar a interface do formulário
        const content = document.getElementById('content');
        if (!content) {
            console.error('Elemento content nao encontrado');
            return;
        }
        
        console.log('Elemento content encontrado');
        content.innerHTML = `
            <div class="page-container">
                <div class="form-header-editable">
                    <div class="editable-title-group">
                        <h1 id="form-title-display" class="editable-title">
                            ${currentForm.name}
                        </h1>
                        <button type="button" class="edit-btn edit-title-btn" title="Editar título">
                            <i class="material-icons">edit</i>
                        </button>
                        <input 
                            type="text" 
                            id="form-title-input" 
                            class="title-input hidden"
                            value="${currentForm.name}"
                        >
                    </div>
                    
                    <div class="editable-description-group">
                        <p id="form-description-display" class="editable-description">
                            ${currentForm.description}
                        </p>
                        <button type="button" class="edit-btn edit-description-btn" title="Editar descrição">
                            <i class="material-icons">edit</i>
                        </button>
                        <textarea 
                            id="form-description-input" 
                            class="description-input hidden"
                            rows="2"
                        >${currentForm.description}</textarea>
                    </div>
                    
                    <div class="form-navigation-tabs">
                        <div class="nav-tabs-left">
                            <button type="button" class="nav-tab-icon active" data-tab="questions" title="Perguntas">
                                <i class="material-icons">quiz</i>
                            </button>
                            <button type="button" class="nav-tab-icon" data-tab="responses" title="Respostas">
                                <i class="material-icons">analytics</i>
                            </button>
                            <button type="button" class="nav-tab-icon" data-tab="preview" title="Visualização">
                                <i class="material-icons">visibility</i>
                            </button>
                        </div>
                        <div class="nav-tabs-right">
                            <button type="button" class="nav-tab-icon" data-tab="settings" title="Configurações">
                                <i class="material-icons">settings</i>
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Canvas de Edição -->
                <div class="form-canvas">
                    <!-- Aba de Perguntas -->
                    <div id="questions-tab" class="tab-content active">
                        <div class="questions-container">
                            <!-- Container vazio - questões criadas dinamicamente via JavaScript -->
                        </div>
                        
                        <!-- FAB - Floating Action Bar -->
                        <div class="floating-action-bar">
                            <button class="fab-btn" data-action="add-question" title="Adicionar pergunta">
                                <i class="material-icons">add</i>
                            </button>
                            <button class="fab-btn" data-action="import-questions" title="Importar perguntas">
                                <i class="material-icons">file_download</i>
                            </button>
                            <button class="fab-btn" data-action="add-image" title="Adicionar imagem">
                                <i class="material-icons">image</i>
                            </button>
                            <button class="fab-btn" data-action="add-video" title="Adicionar vídeo">
                                <i class="material-icons">videocam</i>
                            </button>
                            <button class="fab-btn" data-action="add-section" title="Adicionar seção">
                                <i class="material-icons">view_agenda</i>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Aba de Respostas -->
                    <div id="responses-tab" class="tab-content">
                        <div class="responses-content">
                            <h3>Respostas</h3>
                            <p>Visualização de respostas será implementada aqui</p>
                        </div>
                    </div>
                    
                    <!-- Aba de Visualização -->
                    <div id="preview-tab" class="tab-content">
                        <div class="preview-container">
                            <div class="preview-content">
                                <div class="form-preview">
                                    <div class="preview-form-header">
                                        <h2 id="preview-title">Título do Formulário</h2>
                                        <p id="preview-description">Descrição do formulário aparecerá aqui</p>
                                    </div>
                                    <div id="preview-questions">
                                        <!-- Questões serão renderizadas aqui -->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Aba de Configurações -->
                    <div id="settings-tab" class="tab-content">
                        <div class="settings-content">
                            <div class="settings-section">
                                <h3>Configurações do Formulário</h3>
                                
                                <div class="setting-group">
                                    <label for="form-time-limit">Tempo limite (minutos)</label>
                                    <input type="number" id="form-time-limit" min="0" max="180" value="30">
                                </div>
                                
                                <div class="setting-group">
                                    <label>
                                        <input type="checkbox" id="shuffle-questions">
                                        Embaralhar perguntas
                                    </label>
                                </div>
                                
                                <div class="setting-group">
                                    <label>
                                        <input type="checkbox" id="show-results" checked>
                                        Mostrar resultados ao finalizar
                                    </label>
                                </div>
                                
                                <div class="setting-group">
                                    <label>
                                        <input type="checkbox" id="allow-retake">
                                        Permitir refazer
                                    </label>
                                </div>
                                
                                <div class="setting-group">
                                    <label>
                                        <input type="checkbox" id="collect-email">
                                        Coletar email
                                    </label>
                                </div>
                                
                                <div class="setting-group">
                                    <h4>Resumo</h4>
                                    <div class="form-summary">
                                        <div class="summary-item">
                                            <span class="summary-label">Tempo estimado:</span>
                                            <span id="estimated-time">5 minutos</span>
                                        </div>
                                        <div class="summary-item">
                                            <span class="summary-label">Perguntas:</span>
                                            <span id="question-count">0</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="question-footer">
                                    <div class="question-actions">
                                        <button class="action-btn" title="Resetar configurações">
                                            <i class="material-icons">refresh</i>
                                        </button>
                                        <button class="action-btn" title="Exportar configurações">
                                            <i class="material-icons">download</i>
                                        </button>
                                        <button class="action-btn more-options" title="Mais opções">
                                            <i class="material-icons">more_vert</i>
                                        </button>
                                    </div>
                                    <div class="question-settings">
                                        <span style="color: var(--md-sys-color-on-surface-variant); font-size: 0.875rem;">
                                            <i class="material-icons" style="font-size: 16px; vertical-align: middle;">info</i>
                                            Configurações aplicadas automaticamente
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        console.log('Configurando form creator...');
        this.setupFormCreator();
        
        // Renderizar questões existentes do formulário
        setTimeout(() => {
            this.renderExistingQuestions();
        }, 100);
    }
    
    /**
     * Configura event listeners para configurações do formulário
     */
    setupFormSettingsListeners() {
        console.log('Configurando listeners das configuracoes...');
        
        // Tempo limite
        const timeLimit = document.getElementById('form-time-limit');
        if (timeLimit) {
            timeLimit.addEventListener('input', () => {
                this.updateFormSummary();
                this.syncFormSettingsToRegistry();
            });
        }
        
        // Embaralhar questões
        const shuffleQuestions = document.getElementById('shuffle-questions');
        if (shuffleQuestions) {
            shuffleQuestions.addEventListener('change', () => {
                this.syncFormSettingsToRegistry();
            });
        }
        
        // Mostrar resultados
        const showResults = document.getElementById('show-results');
        if (showResults) {
            showResults.addEventListener('change', () => {
                this.syncFormSettingsToRegistry();
            });
        }
        
        // Permitir refazer
        const allowRetake = document.getElementById('allow-retake');
        if (allowRetake) {
            allowRetake.addEventListener('change', () => {
                this.syncFormSettingsToRegistry();
            });
        }
        
        // Coletar email
        const collectEmail = document.getElementById('collect-email');
        if (collectEmail) {
            collectEmail.addEventListener('change', () => {
                this.syncFormSettingsToRegistry();
            });
        }
        
        console.log('Listeners das configuracoes configurados');
    }
    
    /**
     * Sincroniza configurações do formulário para o formRegistry
     */
    syncFormSettingsToRegistry() {
        const currentFormRegistry = this.getCurrentFormRegistry();
        if (!currentFormRegistry) return;
        
        // Obter valores atuais dos campos
        const timeLimit = document.getElementById('form-time-limit')?.value;
        const shuffleQuestions = document.getElementById('shuffle-questions')?.checked;
        const showResults = document.getElementById('show-results')?.checked;
        const allowRetake = document.getElementById('allow-retake')?.checked;
        const collectEmail = document.getElementById('collect-email')?.checked;
        
        // Atualizar settings no formRegistry
        currentFormRegistry.settings = {
            timeLimit: timeLimit ? parseInt(timeLimit) : 30,
            shuffleQuestions: !!shuffleQuestions,
            showResults: !!showResults,
            allowRetake: !!allowRetake,
            collectEmail: !!collectEmail
        };
        
        // Atualizar data de modificação
        currentFormRegistry.modified = new Date();
        
        console.log('Configuracoes sincronizadas para o registry');
        
        // Auto-save
        this.autoSaveForm();
    }
    
    setupFormCreator() {
        
        // Verificar se os elementos do formulário existem
        const formCreator = document.getElementById('form-creator');
        const timeLimit = document.getElementById('form-time-limit');
        const shuffleQuestions = document.getElementById('shuffle-questions');
        const showResults = document.getElementById('show-results');
        const allowRetake = document.getElementById('allow-retake');
        
        console.log('Elementos encontrados:', {
            formCreator: !!formCreator,
            timeLimit: !!timeLimit,
            shuffleQuestions: !!shuffleQuestions,
            showResults: !!showResults,
            allowRetake: !!allowRetake
        });
        
        // Event listeners para configurações do formulário
        this.setupFormSettingsListeners();
        
        // Atualizar tempo estimado
        this.updateFormSummary();
        
        // Configurar event listeners após carregar o HTML
        this.setupFormEditing();
    }
    
    setupFormEditing() {
        console.log('Configurando event listeners para edicao inline...');
        
        // Event listeners para título
        const titleDisplay = document.getElementById('form-title-display');
        const titleEditBtn = document.querySelector('.edit-title-btn');
        const titleInput = document.getElementById('form-title-input');
        
        if (titleDisplay && titleEditBtn && titleInput) {
            titleDisplay.addEventListener('click', () => this.editTitle());
            titleEditBtn.addEventListener('click', () => this.editTitle());
            titleInput.addEventListener('blur', () => this.saveTitle());
            titleInput.addEventListener('keydown', (e) => this.handleTitleKeydown(e));
            console.log('Event listeners do titulo configurados');
        } else {
            console.error('Elementos do titulo nao encontrados:', { titleDisplay, titleEditBtn, titleInput });
        }
        
        // Event listeners para descrição
        const descDisplay = document.getElementById('form-description-display');
        const descEditBtn = document.querySelector('.edit-description-btn');
        const descInput = document.getElementById('form-description-input');
        
        if (descDisplay && descEditBtn && descInput) {
            descDisplay.addEventListener('click', () => this.editDescription());
            descEditBtn.addEventListener('click', () => this.editDescription());
            descInput.addEventListener('blur', () => this.saveDescription());
            descInput.addEventListener('keydown', (e) => this.handleDescriptionKeydown(e));
            console.log('Event listeners da descricao configurados');
        } else {
            console.error('Elementos da descricao nao encontrados:', { descDisplay, descEditBtn, descInput });
        }
        
        // Event listeners para navegação entre abas
        const navTabs = document.querySelectorAll('.nav-tab, .nav-tab-icon');
        const tabContents = document.querySelectorAll('.tab-content');
        
        navTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetTab = tab.dataset.tab;
                
                // Remover active de todas as abas
                navTabs.forEach(t => t.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Ativar aba clicada
                tab.classList.add('active');
                document.getElementById(`${targetTab}-tab`).classList.add('active');
                
                // Se for a aba preview, renderizar o formulário e esconder elementos de edição
                if (targetTab === 'preview') {
                    this.hideEditingElements();
                    setTimeout(() => {
                        this.renderFormPreview();
                    }, 100); // Pequeno delay para garantir que a aba esteja visível
                } else {
                    // Mostrar elementos de edição nas outras abas
                    this.showEditingElements();
                }
                
                console.log(`Aba ${targetTab} ativada`);
            });
        });
        
        // Event listeners para FAB buttons
        const fabBtns = document.querySelectorAll('.fab-btn');
        fabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                this.handleFabAction(action);
            });
        });
        
        // Event listeners para blocos de pergunta
        this.setupQuestionBlocks();
        
        // Renderizar questões existentes ou inicializar questão padrão
        this.renderFormContent();
        
        console.log('Event listeners da interface configurados');
    }
    
    renderFormContent() {
        console.log('Renderizando conteudo do formulario...');
        
        // Verificar se existe formulário ativo com questões
        if (this.currentFormId && this.questionsRegistry && this.questionsRegistry.length > 0) {
            console.log(`Formulario ativo encontrado: ${this.currentFormId} com ${this.questionsRegistry.length} questoes`);
            this.renderExistingQuestions();
            this.updateFormMetadataInUI();
        } else {
            console.log('Nenhum formulario ativo ou sem questoes, inicializando questao padrao');
            this.initializeDefaultQuestion();
        }
    }
    
    renderExistingQuestions() {
        console.log('Renderizando questoes existentes...');
        
        const questionsContainer = document.querySelector('.questions-container');
        if (!questionsContainer) {
            console.error('Container de questoes nao encontrado');
            return;
        }
        
        // Limpar container
        questionsContainer.innerHTML = '';
        
        // Verificar se há questões válidas
        if (!this.questionsRegistry || this.questionsRegistry.length === 0) {
            console.log('Nenhuma questao encontrada no registry');
            return;
        }
        
        // Renderizar cada questão existente
        this.questionsRegistry.forEach((question, index) => {
            if (!question || !question.id) {
                console.error(`Questao invalida no index ${index}`);
                return;
            }
            
            this.createQuestionHTML(question, index);
            
            setTimeout(() => {
                if (question.itensRegistry && question.itensRegistry.length > 0) {
                    this.renderQuestionOptions(question);
                }
                // Atualizar statement display com markdown
                this.updateStatementDisplay(question.id);
                
                // Re-renderizar justificativa compacta se existir
                if (question.justificativa && question.justificativa.trim()) {
                    const questionBlock = document.querySelector(`[data-question-id="${question.id}"]`);
                    if (questionBlock) {
                        const existingCompact = questionBlock.querySelector('.justification-compact');
                        if (!existingCompact) {
                            this.renderJustificationCompact(questionBlock, question.justificativa);
                            this.disableCorrectOptionCheck(questionBlock);
                        }
                    }
                }
            }, 100); // Aumentei o delay para garantir que o DOM esteja pronto
        });
        
        console.log(`${this.questionsRegistry.length} questoes renderizadas`);
    }
    
    updateFormMetadataInUI() {
        console.log('Atualizando metadados do formulario na interface...');
        
        const formRegistry = this.getCurrentFormRegistry();
        if (!formRegistry) return;
        
        // Atualizar título
        const titleDisplay = document.getElementById('form-title-display');
        const titleInput = document.getElementById('form-title-input');
        if (titleDisplay && titleInput) {
            titleDisplay.textContent = formRegistry.name;
            titleInput.value = formRegistry.name;
        }
        
        // Atualizar descrição
        const descDisplay = document.getElementById('form-description-display');
        const descInput = document.getElementById('form-description-input');
        if (descDisplay && descInput) {
            descDisplay.textContent = formRegistry.description;
            descInput.value = formRegistry.description;
        }
        
        // Atualizar configurações se existirem
        if (formRegistry.settings) {
            const timeLimit = document.getElementById('form-time-limit');
            if (timeLimit && formRegistry.settings.timeLimit !== undefined) {
                timeLimit.value = formRegistry.settings.timeLimit;
            }
            
            const shuffleQuestions = document.getElementById('shuffle-questions');
            if (shuffleQuestions && formRegistry.settings.shuffleQuestions !== undefined) {
                shuffleQuestions.checked = formRegistry.settings.shuffleQuestions;
            }
            
            const showResults = document.getElementById('show-results');
            if (showResults && formRegistry.settings.showResults !== undefined) {
                showResults.checked = formRegistry.settings.showResults;
            }
            
            const allowRetake = document.getElementById('allow-retake');
            if (allowRetake && formRegistry.settings.allowRetake !== undefined) {
                allowRetake.checked = formRegistry.settings.allowRetake;
            }
            
            const collectEmail = document.getElementById('collect-email');
            if (collectEmail && formRegistry.settings.collectEmail !== undefined) {
                collectEmail.checked = formRegistry.settings.collectEmail;
            }
            
            console.log('Configuracoes restauradas do registry');
        }
        
        console.log('Metadados atualizados');
    }
    
    initializeDefaultQuestion() {
        console.log('Inicializando questao padrao...');
        
        // Encontrar a questão padrão no template
        const defaultQuestionBlock = document.querySelector('[data-question-id="1"]');
        if (!defaultQuestionBlock) {
            console.log('Questao padrao nao encontrada');
            return;
        }
        
        // Criar dados da questão padrão com 2 opções iniciais
        const defaultQuestion = {
            id: '1',
            type: 'multiple-choice',
            itensRegistry: [
                { 
                    id: 'a', 
                    text: 'opcao 1', 
                    isCorrect: false,
                    position: 0,
                    order: 1,
                    isOtherItem: false
                },
                { 
                    id: 'b', 
                    text: 'outro item', 
                    isCorrect: false,
                    position: 1,
                    order: 2,
                    isOtherItem: true
                }
            ]
        };
        
        // Renderizar as opções usando o sistema modular existente
        this.renderQuestionOptions(defaultQuestion);
        
        console.log('Questao padrao inicializada');
    }
    
    updateFormSummary() {
        const timeLimit = document.getElementById('form-time-limit')?.value;
        const estimatedTimeSpan = document.getElementById('estimated-time');
        
        if (estimatedTimeSpan) {
            if (timeLimit && timeLimit > 0) {
                estimatedTimeSpan.textContent = `Tempo estimado: ${timeLimit} minutos`;
            } else {
                estimatedTimeSpan.textContent = 'Tempo estimado: Sem limite';
            }
        }
    }
    
    autoSaveForm() {
        // Salvar automaticamente no localStorage
        const formData = this.getFormData();
        if (formData.title || formData.description) {
            localStorage.setItem('kyrion_form_draft', JSON.stringify(formData));
            console.log('Formulario salvo automaticamente');
        }
    }
    
    getFormData() {
        const formData = {
            id: Date.now().toString(),
            title: document.getElementById('form-title-display')?.textContent || 'Novo Formulário',
            description: document.getElementById('form-description-display')?.textContent || 'Crie um formulário com questões personalizadas',
            timeLimit: document.getElementById('form-time-limit')?.value || null,
            shuffleQuestions: document.getElementById('shuffle-questions')?.checked || false,
            showResults: document.getElementById('show-results')?.checked || true,
            allowRetake: document.getElementById('allow-retake')?.checked || false,
            questions: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            status: 'draft'
        };
        
        console.log('FormData coletado');
        return formData;
    }
    
    saveFormDraft() {
        const formData = this.getFormData();
        
        if (!formData.title.trim()) {
            alert('Por favor, insira um titulo para o formulario');
            return;
        }
        
        // Salvar no localStorage
        const savedForms = JSON.parse(localStorage.getItem('kyrion_forms') || '[]');
        const existingIndex = savedForms.findIndex(form => form.id === formData.id);
        
        if (existingIndex >= 0) {
            savedForms[existingIndex] = formData;
        } else {
            savedForms.push(formData);
        }
        
        localStorage.setItem('kyrion_forms', JSON.stringify(savedForms));
        
        // Feedback visual
        const btn = event.target;
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="material-icons">check</i> Salvo!';
        btn.style.backgroundColor = 'var(--md-sys-color-tertiary)';
        
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.style.backgroundColor = '';
        }, 2000);
        
        console.log('Rascunho salvo com sucesso');
    }
    
    proceedToQuestions() {
        console.log('proceedToQuestions chamado');
        const formData = this.getFormData();
        console.log('Auto-save executado');
        
        if (!formData.title.trim() || formData.title === 'Novo Formulário') {
            alert('Por favor, edite o titulo do formulario clicando no icone de caneta');
            return;
        }
        
        // Salvar os dados básicos
        this.saveFormDraft();
        
        // Armazenar dados do formulário atual para o editor de questões
        sessionStorage.setItem('current_form', JSON.stringify(formData));
        console.log('Dados salvos na sessao');
        
        // Salvar e permanecer na mesma tela
        console.log('Formulario salvo com sucesso');
    }

    loadFormsAnswered() {
        const content = document.getElementById('content');
        if (!content) return;
        
        content.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1>Formulários Respondidos</h1>
                    <p>Visualize todos os formulários que foram respondidos</p>
                </div>
                
                <div class="empty-state">
                    <div class="empty-icon">
                        <i class="material-icons">assignment_turned_in</i>
                    </div>
                    <h3>Nenhum formulário respondido ainda</h3>
                    <p>Os formulários respondidos aparecerão aqui quando houver respostas</p>
                </div>
            </div>
        `;
    }
    
    loadFormsGrading() {
        const content = document.getElementById('content');
        if (!content) return;
        
        content.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1>Formulários em Correção</h1>
                    <p>Gerencie formulários que estão aguardando correção</p>
                </div>
                
                <div class="empty-state">
                    <div class="empty-icon">
                        <i class="material-icons">grading</i>
                    </div>
                    <h3>Nenhum formulário em correção</h3>
                    <p>Formulários que necessitam de correção aparecerão aqui</p>
                </div>
            </div>
        `;
    }

    loadResponses() {
        const content = document.getElementById('content');
        if (!content) return;
        
        content.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1>Respostas</h1>
                    <p>Analise as respostas dos seus formulários</p>
                </div>
                
                <div class="empty-state">
                    <div class="empty-icon">
                        <i class="material-icons">analytics</i>
                    </div>
                    <h3>Nenhuma resposta ainda</h3>
                    <p>As respostas aparecerão aqui quando seus formulários forem respondidos</p>
                </div>
            </div>
        `;
    }
    
    
    /**
     * Inicializa a questão padrão do template HTML
     */
    initializeDefaultQuestion() {
        const container = document.querySelector('.questions-container');
        if (!container) return;
        
        this.createMultipleChoiceQuestion('initial');
    }
    
    /**
     * Cria bloco de imagem
     */
    createImageBlock() {
        try {
            const insertPosition = this.determineInsertPosition('fab');
            const imageId = `i_${Date.now()}`;
            const imageNumber = (this.questionsRegistry ? this.questionsRegistry.length : 0) + 1;
            
            const imageData = {
                id: imageId,
                number: imageNumber,
                type: 'image',
                title: this.generateQuestionTitle(imageNumber, 'image'),
                description: '',
                imageUrl: '',
                itensRegistry: []
            };
            
            // Garantir que o registry está inicializado
            if (!this.questionsRegistry) {
                console.warn('Registry nao estava inicializado, criando array vazio');
                this.questionsRegistry = [];
            }
            
            // Remover foco de todas as questões
            this.questionsRegistry.forEach(q => q.focus = false);
            
            // Calcular ordem da nova imagem
            const newOrder = this.calculateQuestionOrder(insertPosition);
            
            this.questionsRegistry.splice(insertPosition, 0, {
                id: imageId,
                type: 'image',
                position: insertPosition,
                order: newOrder,
                focus: true,
                enunciado: '',
                justificativa: '',
                rubrica: '',
                selecionado: false,
                rubricaform: false,
                description: '',
                imageUrl: '',
                itensRegistry: [],
                domElement: null
            });
            
            // Atualizar posições e ordens de todas as questões
            this.updateQuestionsOrder();
            
            // Dados da imagem criados e renderizados no DOM
            this.createQuestionHTML(imageData, insertPosition);
            
            // Atualizar display se existir
            this.updateStatementDisplay(imageData.id);
            
            // Atualizar metadados do formulário
            this.updateCurrentFormMetadata();
            this.renderFormCards();
            
            return imageData;
            
        } catch (error) {
            console.error('Erro na criação da imagem:', error);
            return null;
        }
    }

    /**
     * Cria bloco de vídeo
     */
    createVideoBlock() {
        try {
            const insertPosition = this.determineInsertPosition('fab');
            const videoId = `v_${Date.now()}`;
            const videoNumber = (this.questionsRegistry ? this.questionsRegistry.length : 0) + 1;
            
            const videoData = {
                id: videoId,
                number: videoNumber,
                type: 'video',
                title: this.generateQuestionTitle(videoNumber, 'video'),
                description: '',
                videoUrl: '',
                itensRegistry: []
            };
            
            // Garantir que o registry está inicializado
            if (!this.questionsRegistry) {
                console.warn('Registry nao estava inicializado, criando array vazio');
                this.questionsRegistry = [];
            }
            
            // Remover foco de todas as questões
            this.questionsRegistry.forEach(q => q.focus = false);
            
            // Calcular ordem do novo vídeo
            const newOrder = this.calculateQuestionOrder(insertPosition);
            
            this.questionsRegistry.splice(insertPosition, 0, {
                id: videoId,
                type: 'video',
                position: insertPosition,
                order: newOrder,
                focus: true,
                enunciado: '',
                justificativa: '',
                rubrica: '',
                selecionado: false,
                rubricaform: false,
                description: '',
                videoUrl: '',
                itensRegistry: [],
                domElement: null
            });
            
            // Atualizar posições e ordens de todas as questões
            this.updateQuestionsOrder();
            
            // Dados do vídeo criados e renderizados no DOM
            this.createQuestionHTML(videoData, insertPosition);
            
            // Atualizar display se existir
            this.updateStatementDisplay(videoData.id);
            
            // Atualizar metadados do formulário
            this.updateCurrentFormMetadata();
            this.renderFormCards();
            
            return videoData;
            
        } catch (error) {
            console.error('Erro na criação do vídeo:', error);
            return null;
        }
    }
    
    /**
     * Cria bloco de seção
     */
    createSectionBlock() {
        try {
            const insertPosition = this.determineInsertPosition('fab');
            const sectionId = `s_${Date.now()}`;
            const sectionNumber = (this.questionsRegistry ? this.questionsRegistry.length : 0) + 1;
            
            const sectionData = {
                id: sectionId,
                number: sectionNumber,
                type: 'section',
                title: this.generateQuestionTitle(sectionNumber, 'section'),
                itensRegistry: []
            };
            
            // Garantir que o registry está inicializado
            if (!this.questionsRegistry) {
                console.warn('Registry nao estava inicializado, criando array vazio');
                this.questionsRegistry = [];
            }
            
            // Remover foco de todas as questões
            this.questionsRegistry.forEach(q => q.focus = false);
            
            // Calcular ordem da nova seção
            const newOrder = this.calculateQuestionOrder(insertPosition);
            
            this.questionsRegistry.splice(insertPosition, 0, {
                id: sectionId,
                type: 'section',
                position: insertPosition,
                order: newOrder,
                focus: true,
                enunciado: '',
                justificativa: '',
                rubrica: '',
                selecionado: false,
                rubricaform: false,
                itensRegistry: [],
                domElement: null
            });
            
            // Atualizar posições e ordens de todas as questões
            this.updateQuestionsOrder();
            
            // Dados da seção criados e renderizados no DOM
            this.createQuestionHTML(sectionData, insertPosition);
            
            // Atualizar display do enunciado se existir
            this.updateStatementDisplay(sectionData.id);
            
            // Atualizar metadados do formulário
            this.updateCurrentFormMetadata();
            this.renderFormCards();
            
            return sectionData;
            
        } catch (error) {
            console.error('Erro na criação da seção:', error);
            return null;
        }
    }
    
    /**
     * Cria questão de múltipla escolha usando estrutura de dados para posicionamento
     * @param {string} context - De onde foi chamada ('default', 'fab', 'button')
     */
    createMultipleChoiceQuestion(context = 'default') {
        try {
            const insertPosition = this.determineInsertPosition(context);
            const questionId = (context === 'default' || context === 'initial') ? '1' : `q_${Date.now()}`;
            const questionNumber = (this.questionsRegistry ? this.questionsRegistry.length : 0) + 1;
            
            const questionData = {
                id: questionId,
                number: questionNumber,
                type: 'multiple-choice',
                title: this.generateQuestionTitle(questionNumber, 'multiple-choice'),
                itensRegistry: [
                    { 
                        id: 'a', 
                        text: 'opcao 1', 
                        isCorrect: false,
                        position: 0,
                        order: 1,
                        isOtherItem: false
                    },
                    { 
                        id: 'b', 
                        text: 'outro item', 
                        isCorrect: false,
                        position: 1,
                        order: 2,
                        isOtherItem: true
                    }
                ],
                required: false,
                points: 1,
                position: insertPosition
            };
            
            // Garantir que o registry está inicializado (não deveria acontecer se initializeFormRegistry foi chamado)
            if (!this.questionsRegistry) {
                console.warn('Registry nao estava inicializado, criando array vazio');
                this.questionsRegistry = [];
            }
            
            // Remover foco de todas as questões
            this.questionsRegistry.forEach(q => q.focus = false);
            
            // Calcular ordem da nova questão
            const newOrder = this.calculateQuestionOrder(insertPosition);
            
            this.questionsRegistry.splice(insertPosition, 0, {
                id: questionId,
                type: 'multiple-choice',
                position: insertPosition,
                order: newOrder,
                focus: true,           // Nova questão sempre recebe foco
                enunciado: '',        // Texto principal da questão (pergunta)
                justificativa: '',     // Texto de justificativa da questão
                rubrica: '',          // Texto da rubrica de avaliação
                selecionado: false,   // Se a questão está selecionada na interface
                rubricaform: false,   // Se o formulário de rubrica está aberto
                itensRegistry: [      // Estrutura de dados dos itens da questão
                    {
                        id: 'a',
                        text: 'opcao 1',
                        isCorrect: false,
                        position: 0,
                        order: 1,
                        isOtherItem: false
                    },
                    {
                        id: 'b',
                        text: 'outro item',
                        isCorrect: false,
                        position: 1,
                        order: 2,
                        isOtherItem: true
                    }
                ],
                domElement: null
            });
            
            // Atualizar posições e ordens de todas as questões
            this.updateQuestionsOrder();
            
            // Dados da questão criados e renderizados no DOM
            this.createQuestionHTML(questionData, insertPosition);
            
            // Usar setTimeout para garantir que o DOM foi atualizado
            setTimeout(() => {
                this.renderQuestionOptions(questionData);
            }, 100);
            
            // Atualizar display do enunciado se existir
            this.updateStatementDisplay(questionData.id);
            
            setTimeout(() => {
                this.focusOnOption('a');
            }, 200);
            
            // Atualizar metadados do formulário
            this.updateCurrentFormMetadata();
            this.renderFormCards();
            
            return questionData;
            
        } catch (error) {
            console.error('Erro na criação da questão:', error);
            return null;
        }
    }
    
    /**
     * Determina onde inserir nova questão baseado no contexto e questão ativa
     * @param {string} context - Contexto da criação
     * @returns {number} Posição de inserção
     */
    determineInsertPosition(context) {
        if (context === 'default' || context === 'initial') {
            return 0;
        }
        
        const selectedQuestion = document.querySelector('.question-block.selected');
        
        if (selectedQuestion) {
            const selectedId = selectedQuestion.dataset.questionId;
            
            if (this.questionsRegistry) {
                const selectedIndex = this.questionsRegistry.findIndex(q => q.id === selectedId);
                
                if (selectedIndex !== -1) {
                    return selectedIndex + 1;
                }
            }
        }
        
        return this.questionsRegistry ? this.questionsRegistry.length : 0;
    }
    
    /**
     * Calcula a ordem (order) para uma nova questão baseada na posição de inserção
     * @param {number} insertPosition - Posição onde será inserida
     * @returns {number} Ordem da nova questão
     */
    calculateQuestionOrder(insertPosition) {
        if (insertPosition === 0) {
            return 1; // Primeira questão sempre é ordem 1
        }
        
        // Nova questão inserida após a questão na posição anterior
        const previousQuestion = this.questionsRegistry[insertPosition - 1];
        return previousQuestion ? previousQuestion.order + 1 : insertPosition + 1;
    }
    
    /**
     * Atualiza as posições e ordens de todas as questões no registry
     * Chamada após inserções, remoções ou reordenações
     */
    updateQuestionsOrder() {
        this.questionsRegistry.forEach((question, index) => {
            question.position = index;
            question.order = index + 1;
            
            // Atualizar título da questão no DOM
            const questionBlock = document.querySelector(`[data-question-id="${question.id}"]`);
            if (questionBlock) {
                this.updateQuestionTitle(questionBlock, question);
            }
        });
        
        console.log('Ordens e titulos de todas as questoes atualizados');
    }
    
    /**
     * Define foco em uma questão específica, removendo de todas as outras
     * @param {string} questionId - ID da questão que deve receber foco
     */
    setQuestionFocus(questionId) {
        this.questionsRegistry.forEach(question => {
            question.focus = (question.id === questionId);
            question.selecionado = (question.id === questionId);
        });
    }

    /**
     * Atualiza o enunciado de uma questão específica
     * @param {string} questionId - ID da questão
     * @param {string} enunciado - Novo enunciado
     */
    setQuestionEnunciado(questionId, enunciado) {
        const question = this.questionsRegistry.find(q => q.id === questionId);
        if (question) {
            question.enunciado = enunciado;
        }
    }

    /**
     * Atualiza a justificativa de uma questão específica
     * @param {string} questionId - ID da questão
     * @param {string} justificativa - Nova justificativa
     */
    setQuestionJustificativa(questionId, justificativa) {
        const question = this.questionsRegistry.find(q => q.id === questionId);
        if (question) {
            question.justificativa = justificativa;
        }
    }

    /**
     * Atualiza a rubrica de uma questão específica
     * @param {string} questionId - ID da questão
     * @param {string} rubrica - Nova rubrica
     */
    setQuestionRubrica(questionId, rubrica) {
        const question = this.questionsRegistry.find(q => q.id === questionId);
        if (question) {
            question.rubrica = rubrica;
        }
    }

    /**
     * Controla se o formulário de rubrica está aberto/fechado
     * @param {string} questionId - ID da questão
     * @param {boolean} isOpen - Se o formulário deve estar aberto
     */
    setQuestionRubricaForm(questionId, isOpen) {
        const question = this.questionsRegistry.find(q => q.id === questionId);
        if (question) {
            question.rubricaform = isOpen;
        }
    }

    /**
     * Obtém dados completos de uma questão do registry
     * @param {string} questionId - ID da questão
     * @returns {Object|null} Dados da questão ou null se não encontrada
     */
    getQuestionFromRegistry(questionId) {
        return this.questionsRegistry.find(q => q.id === questionId) || null;
    }

    /**
     * Obtém todas as questões que estão com foco
     * @returns {Array} Array das questões com foco
     */
    getFocusedQuestions() {
        return this.questionsRegistry.filter(q => q.focus);
    }

    /**
     * Obtém todas as questões selecionadas
     * @returns {Array} Array das questões selecionadas
     */
    getSelectedQuestions() {
        return this.questionsRegistry.filter(q => q.selecionado);
    }
    
    /**
     * Cria HTML da questão e adiciona ao DOM na posição específica
     * @param {Object} questionData - Dados da questão
     * @param {number} insertPosition - Posição onde inserir
     */
    createQuestionHTML(questionData, insertPosition) {
        const container = document.querySelector('.questions-container');
        if (!container) return;
        
        const existingQuestions = container.querySelectorAll('.question-block');
        
        let questionHTML;
        
        if (questionData.type === 'section') {
            // HTML específico para seções - só header com título e footer, sem select box
            questionHTML = `
                <div class="question-block selected section-block" data-question-id="${questionData.id}">
                    <div class="question-header section-header">
                        <input type="text" class="question-title" placeholder="Título da seção" value="${questionData.title}">
                    </div>
                    
                    <div class="question-footer">
                        <div class="question-actions">
                            <button type="button" class="btn-icon" title="Duplicar">
                                <i class="material-icons">content_copy</i>
                            </button>
                            <button type="button" class="btn-icon copy-question" title="Copiar para clipboard">
                                <i class="material-icons">file_copy</i>
                            </button>
                            <button type="button" class="btn-icon" title="Excluir">
                                <i class="material-icons">delete</i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        } else if (questionData.type === 'image') {
            // HTML específico para imagens - header, corpo e footer
            questionHTML = `
                <div class="question-block selected image-block" data-question-id="${questionData.id}">
                    <div class="question-header">
                        <input type="text" class="question-title" placeholder="Título da imagem" value="${questionData.title}">
                        <select class="question-type">
                            <option value="multiple-choice">Múltipla escolha</option>
                            <option value="checkboxes">Caixas de seleção</option>
                            <option value="short-answer">Resposta Curta</option>
                            <option value="paragraph">Resposta Longa - Texto</option>
                            <option value="long-answer">Resposta Longa - Matemática e Código</option>
                            <option value="file-upload">Upload de arquivo</option>
                            <option value="section">Seção</option>
                            <option value="video">Vídeo</option>
                            <option value="image" selected>Imagem</option>
                        </select>
                    </div>
                    
                    <div class="question-content image-content">
                        <div class="image-fields">
                            <div class="field-group">
                                <label for="image-description-${questionData.id}">Descrição:</label>
                                <textarea id="image-description-${questionData.id}" class="image-description" 
                                         placeholder="Digite uma descrição para a imagem...">${questionData.description || ''}</textarea>
                            </div>
                            <div class="field-group">
                                <label for="image-url-${questionData.id}">URL da imagem:</label>
                                <input type="url" id="image-url-${questionData.id}" class="image-url" 
                                       placeholder="https://exemplo.com/imagem.jpg" value="${questionData.imageUrl || ''}" />
                            </div>
                            <div class="field-group">
                                <div class="image-preview placeholder" id="image-preview-${questionData.id}">
                                    <div class="preview-placeholder">
                                        <i class="material-icons">image</i>
                                        <span>Preview da imagem aparecerá aqui após inserir a URL</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="question-footer">
                        <div class="question-actions">
                            <button type="button" class="btn-icon" title="Duplicar">
                                <i class="material-icons">content_copy</i>
                            </button>
                            <button type="button" class="btn-icon copy-question" title="Copiar para clipboard">
                                <i class="material-icons">file_copy</i>
                            </button>
                            <button type="button" class="btn-icon" title="Excluir">
                                <i class="material-icons">delete</i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        } else if (questionData.type === 'video') {
            // HTML específico para vídeos - header, corpo e footer
            questionHTML = `
                <div class="question-block selected video-block" data-question-id="${questionData.id}">
                    <div class="question-header">
                        <input type="text" class="question-title" placeholder="Título do vídeo" value="${questionData.title}">
                        <select class="question-type">
                            <option value="multiple-choice">Múltipla escolha</option>
                            <option value="checkboxes">Caixas de seleção</option>
                            <option value="short-answer">Resposta Curta</option>
                            <option value="paragraph">Resposta Longa - Texto</option>
                            <option value="long-answer">Resposta Longa - Matemática e Código</option>
                            <option value="file-upload">Upload de arquivo</option>
                            <option value="section">Seção</option>
                            <option value="video" selected>Vídeo</option>
                        </select>
                    </div>
                    
                    <div class="question-content video-content">
                        <div class="video-fields">
                            <div class="field-group">
                                <label for="video-description-${questionData.id}">Descrição:</label>
                                <textarea id="video-description-${questionData.id}" class="video-description" 
                                         placeholder="Digite uma descrição para o vídeo...">${questionData.description || ''}</textarea>
                            </div>
                            <div class="field-group">
                                <label for="video-url-${questionData.id}">Link do vídeo:</label>
                                <input type="url" id="video-url-${questionData.id}" class="video-url" 
                                       placeholder="https://www.youtube.com/watch?v=..." value="${questionData.videoUrl || ''}" />
                            </div>
                            <div class="field-group">
                                <div class="video-preview placeholder" id="video-preview-${questionData.id}">
                                    <div class="preview-placeholder">
                                        <i class="material-icons">videocam</i>
                                        <span>Preview do vídeo aparecerá aqui após inserir o link</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="question-footer">
                        <div class="question-actions">
                            <button type="button" class="btn-icon" title="Duplicar">
                                <i class="material-icons">content_copy</i>
                            </button>
                            <button type="button" class="btn-icon copy-question" title="Copiar para clipboard">
                                <i class="material-icons">file_copy</i>
                            </button>
                            <button type="button" class="btn-icon" title="Excluir">
                                <i class="material-icons">delete</i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // HTML padrão para questões normais
            questionHTML = `
                <div class="question-block selected" data-question-id="${questionData.id}">
                    <div class="question-header">
                        <input type="text" class="question-title" placeholder="Pergunta sem título" value="${questionData.title}">
                        <select class="question-type">
                            <option value="multiple-choice" ${questionData.type === 'multiple-choice' ? 'selected' : ''}>Múltipla escolha</option>
                            <option value="checkboxes" ${questionData.type === 'checkboxes' ? 'selected' : ''}>Caixas de seleção</option>
                            <option value="short-answer" ${questionData.type === 'short-answer' ? 'selected' : ''}>Resposta Curta</option>
                            <option value="paragraph" ${questionData.type === 'paragraph' ? 'selected' : ''}>Resposta Longa - Texto</option>
                            <option value="long-answer" ${questionData.type === 'long-answer' ? 'selected' : ''}>Resposta Longa - Matemática e Código</option>
                            <option value="file-upload" ${questionData.type === 'file-upload' ? 'selected' : ''}>Upload de arquivo</option>
                            <option value="section">Seção</option>
                            <option value="video">Vídeo</option>
                            <option value="image">Imagem</option>
                        </select>
                    </div>
                    
                    <div class="question-content">
                        <div class="question-statement" data-question-id="${questionData.id}">
                            <label class="statement-label" for="statement-${questionData.id}">Enunciado da Questão (GFM):</label>
                            <div class="statement-display" onclick="editQuestionStatement('${questionData.id}')">
                                Clique para adicionar o enunciado da questão
                            </div>
                            <textarea class="statement-input hidden" id="statement-${questionData.id}" 
                                    placeholder="Digite o enunciado da questão..."
                                    onblur="saveQuestionStatement('${questionData.id}')"
                                    onkeydown="handleStatementKeydown(event, '${questionData.id}')"></textarea>
                        </div>
                    </div>
                    
                    <div class="question-options">
                        <!-- Opções serão renderizadas via renderQuestionOptions -->
                    </div>
                    
                    <div class="question-footer">
                        <div class="question-actions">
                            <button type="button" class="btn-icon" title="Duplicar">
                                <i class="material-icons">content_copy</i>
                            </button>
                            <button type="button" class="btn-icon copy-question" title="Copiar para clipboard">
                                <i class="material-icons">file_copy</i>
                            </button>
                            <button type="button" class="btn-icon" title="Excluir">
                                <i class="material-icons">delete</i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Remover seleção dos outros blocos
        existingQuestions.forEach(block => block.classList.remove('selected'));
        
        // Inserir na posição específica
        if (insertPosition === 0 || existingQuestions.length === 0) {
            container.insertAdjacentHTML('afterbegin', questionHTML);
        } else if (insertPosition >= existingQuestions.length) {
            container.insertAdjacentHTML('beforeend', questionHTML);
        } else {
            const targetQuestion = existingQuestions[insertPosition - 1];
            targetQuestion.insertAdjacentHTML('afterend', questionHTML);
        }
        
        // Atualizar referência na estrutura de dados
        const newElement = container.querySelector(`[data-question-id="${questionData.id}"]`);
        const registryItem = this.questionsRegistry.find(q => q.id === questionData.id);
        if (registryItem) {
            registryItem.domElement = newElement;
        }
        
        // Atualizar o statement display com conteúdo do registry (incluindo markdown)
        setTimeout(() => {
            this.updateStatementDisplay(questionData.id);
        }, 10);
    }
    
    /**
     * Função ÚNICA para gerar HTML de questão de múltipla escolha
     * @param {string} questionId - ID da questão
     * @param {number} questionNumber - Número da questão
     * @param {string} questionType - Tipo da questão
     * @returns {string} HTML da questão
     */
    generateMultipleChoiceHTML(questionId, questionNumber = 1, questionType = 'multiple-choice') {
        const title = this.generateQuestionTitle(questionNumber, questionType);
        return `
            <div class="question-block selected" data-question-id="${questionId}">
                <div class="question-header">
                    <input type="text" class="question-title" placeholder="Pergunta sem título" value="${title}">
                    <select class="question-type">
                        <option value="multiple-choice" selected>Múltipla escolha</option>
                        <option value="checkboxes">Caixas de seleção</option>
                        <option value="short-answer">Resposta Curta</option>
                        <option value="paragraph">Resposta Longa - Texto</option>
                        <option value="long-answer">Resposta Longa - Matemática e Código</option>
                        <option value="file-upload">Upload de arquivo</option>
                        <option value="section">Seção</option>
                        <option value="video">Vídeo</option>
                        <option value="image">Imagem</option>
                    </select>
                </div>
                
                <!-- Container de opções será preenchido via renderQuestionOptions() -->
                <div class="question-options">
                    <!-- Opções geradas dinamicamente pela nova UX -->
                </div>
                
                <div class="question-footer">
                    <div class="question-actions">
                        <button type="button" class="btn-icon" title="Duplicar">
                            <i class="material-icons">content_copy</i>
                        </button>
                        <button type="button" class="btn-icon copy-question" title="Copiar para clipboard">
                            <i class="material-icons">file_copy</i>
                        </button>
                        <button type="button" class="btn-icon" title="Excluir">
                            <i class="material-icons">delete</i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }
    

    // Métodos de edição inline
    editTitle() {
        console.log('editTitle chamado');
        const display = document.getElementById('form-title-display');
        const input = document.getElementById('form-title-input');
        
        
        if (display && input) {
            input.value = display.textContent;
            display.classList.add('hidden');
            input.classList.remove('hidden');
            input.focus();
            input.select();
            console.log('Titulo em modo de edicao');
        } else {
            console.error('Elementos nao encontrados para edicao do titulo');
        }
    }
    
    saveTitle() {
        const display = document.getElementById('form-title-display');
        const input = document.getElementById('form-title-input');
        
        if (display && input) {
            const newTitle = input.value.trim() || 'Novo Formulário';
            display.textContent = newTitle;
            
            // Atualizar o formRegistry com o novo título
            const currentFormRegistry = this.getCurrentFormRegistry();
            if (currentFormRegistry) {
                currentFormRegistry.name = newTitle;
                currentFormRegistry.modified = new Date();
                
                // Atualizar cards na sidebar
                this.renderFormCards();
                
                console.log(`Titulo atualizado no registry: ${newTitle}`);
            }
            
            display.classList.remove('hidden');
            input.classList.add('hidden');
            
            this.autoSaveForm();
            console.log('Titulo salvo');
        }
    }
    
    handleTitleKeydown(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            this.saveTitle();
        } else if (event.key === 'Escape') {
            event.preventDefault();
            this.cancelTitleEdit();
        }
    }
    
    cancelTitleEdit() {
        const display = document.getElementById('form-title-display');
        const input = document.getElementById('form-title-input');
        
        if (display && input) {
            // Restaurar valor original
            input.value = display.textContent;
            display.classList.remove('hidden');
            input.classList.add('hidden');
        }
    }
    
    editDescription() {
        console.log('editDescription chamado');
        const display = document.getElementById('form-description-display');
        const input = document.getElementById('form-description-input');
        
        
        if (display && input) {
            input.value = display.textContent;
            display.classList.add('hidden');
            input.classList.remove('hidden');
            input.focus();
            input.select();
            console.log('Descricao em modo de edicao');
        } else {
            console.error('Elementos nao encontrados para edicao da descricao');
        }
    }
    
    saveDescription() {
        const display = document.getElementById('form-description-display');
        const input = document.getElementById('form-description-input');
        
        if (display && input) {
            const newDescription = input.value.trim() || 'Crie um formulário com questões personalizadas';
            display.textContent = newDescription;
            
            // Atualizar o formRegistry com a nova descrição
            const currentFormRegistry = this.getCurrentFormRegistry();
            if (currentFormRegistry) {
                currentFormRegistry.description = newDescription;
                currentFormRegistry.modified = new Date();
                
                console.log(`Descricao atualizada no registry: ${newDescription}`);
            }
            
            display.classList.remove('hidden');
            input.classList.add('hidden');
            
            this.autoSaveForm();
            console.log('Descricao salva');
        }
    }
    
    handleDescriptionKeydown(event) {
        if (event.key === 'Enter' && event.ctrlKey) {
            event.preventDefault();
            this.saveDescription();
        } else if (event.key === 'Escape') {
            event.preventDefault();
            this.cancelDescriptionEdit();
        }
    }
    
    cancelDescriptionEdit() {
        const display = document.getElementById('form-description-display');
        const input = document.getElementById('form-description-input');
        
        if (display && input) {
            // Restaurar valor original
            input.value = display.textContent;
            display.classList.remove('hidden');
            input.classList.add('hidden');
        }
    }

    toggleConfiguration() {
        console.log('Alternando visibilidade das configuracoes...');
        const formCreator = document.getElementById('form-creator');
        const configBtn = document.getElementById('show-config-btn');
        
        if (formCreator && configBtn) {
            const isHidden = formCreator.classList.contains('hidden');
            
            if (isHidden) {
                // Mostrar configurações
                formCreator.classList.remove('hidden');
                configBtn.innerHTML = '<i class="material-icons">expand_less</i>Ocultar Configurações';
                console.log('Configuracoes mostradas');
            } else {
                // Ocultar configurações
                formCreator.classList.add('hidden');
                configBtn.innerHTML = '<i class="material-icons">settings</i>Configurações';
                console.log('Configuracoes ocultadas');
            }
        } else {
            console.error('Elementos nao encontrados para toggle:', { formCreator, configBtn });
        }
    }

    handleFabAction(action) {
        console.log(`Acao FAB: ${action}`);
        
        switch (action) {
            case 'add-question':
                this.addNewQuestion();
                break;
            case 'import-questions':
                console.log('Importar perguntas - em desenvolvimento');
                break;
            case 'add-image':
                this.createImageBlock();
                break;
            case 'add-video':
                this.createVideoBlock();
                break;
            case 'add-section':
                this.createSectionBlock();
                break;
            default:
                console.log(`Acao desconhecida: ${action}`);
        }
    }
    
    setupQuestionBlocks() {
        console.log('Configurando blocos de pergunta...');
        
        // Usar delegação de eventos no container principal para evitar múltiplos listeners
        const container = document.querySelector('.questions-container');
        if (!container) {
            console.log('Container .questions-container nao encontrado');
            return;
        }
        
        // Remover listeners existentes se houver
        if (container.hasAttribute('data-listeners-setup')) {
            console.log('Listeners ja configurados para este container');
            return; // Já configurado
        }
        
        // Marcar como configurado
        container.setAttribute('data-listeners-setup', 'true');
        
        // Delegação de eventos para todos os elementos dentro do container
        container.addEventListener('click', (e) => {
            const target = e.target;
            const closestBtn = target.closest('button');
            const questionBlock = target.closest('.question-block');
            
            // Múltipla Escolha: Clique em campo de "outro item" (nova UX inteligente) - VERIFICAR ANTES DA SELEÇÃO GERAL
            if (target.classList.contains('option-text') && questionBlock) {
                const optionItem = target.closest('.option-item');
                const isOtherItem = optionItem?.dataset.isOther === 'true';
                
                if (isOtherItem) {
                    e.stopPropagation();
                    const questionId = questionBlock.dataset.questionId;
                    
                    // Buscar dados do registry (fonte principal)
                    const question = this.getQuestionFromRegistry(questionId);
                    
                    if (question) {
                        this.handleOtherItemClick(optionItem, question);
                    }
                    return;
                }
                
                // Se não é "outro item", continua com lógica normal de seleção abaixo
            }
            
            // Checkboxes: Clique no preview de "outro item"
            if (target.classList.contains('option-preview-text') && questionBlock) {
                const optionItem = target.closest('.option-item');
                const isOtherItem = optionItem?.dataset.isOther === 'true';
                
                if (isOtherItem) {
                    e.stopPropagation();
                    const questionId = questionBlock.dataset.questionId;
                    const question = this.getQuestionFromRegistry(questionId);
                    
                    if (question) {
                        this.handleOtherItemClick(optionItem, question);
                    }
                    return;
                }
            }
            
            // Seleção de blocos de pergunta
            if (questionBlock && !closestBtn) {
                // Remover seleção de outros blocos
                document.querySelectorAll('.question-block').forEach(b => b.classList.remove('selected'));
                // Selecionar bloco atual
                questionBlock.classList.add('selected');
                
                // Atualizar foco no registry
                const questionId = questionBlock.dataset.questionId;
                if (questionId && this.questionsRegistry) {
                    this.setQuestionFocus(questionId);
                }
                
                return;
            }
            
            // Botão de marcar opção como correta
            if (target.closest('.check-option')) {
                e.stopPropagation();
                const optionItem = target.closest('.option-item');
                this.toggleCorrectOption(optionItem);
                return;
            }
            
            
            // Botão de remover opção
            if (target.closest('.remove-option')) {
                e.stopPropagation();
                const optionItem = target.closest('.option-item');
                this.removeOption(optionItem);
                return;
            }
            
            // Botão de duplicar
            if (closestBtn && closestBtn.getAttribute('title') === 'Duplicar' && questionBlock) {
                e.stopPropagation();
                console.log('Duplicar clicado via delegacao de eventos');
                this.duplicateQuestionBlock(questionBlock);
                return;
            }
            
            // Botão de copiar para clipboard
            if (closestBtn && closestBtn.getAttribute('title') === 'Copiar para clipboard' && questionBlock) {
                e.stopPropagation();
                console.log('Copiar para clipboard clicado via delegacao de eventos');
                this.copyQuestionToClipboard(questionBlock);
                return;
            }
            
            // Botão de excluir
            if (closestBtn && closestBtn.getAttribute('title') === 'Excluir' && questionBlock) {
                e.stopPropagation();
                const questionId = questionBlock.dataset.questionId;
                this.deleteQuestionBlock(questionBlock, questionId);
                return;
            }
            
            // Botão de salvar justificativa
            if (target.closest('.justification-save')) {
                e.stopPropagation();
                this.saveJustification(questionBlock);
                return;
            }
            
            // Botão de cancelar justificativa
            if (target.closest('.justification-cancel')) {
                e.stopPropagation();
                this.cancelJustification(questionBlock);
                return;
            }
            
            // Botão de editar justificativa (modo compacto)
            if (target.closest('.justification-edit')) {
                e.stopPropagation();
                this.editJustification(questionBlock);
                return;
            }
            
            // Botão de remover justificativa (modo compacto)
            if (target.closest('.justification-remove')) {
                e.stopPropagation();
                this.removeJustification(questionBlock);
                return;
            }
        });
        
        // Event listeners para mudança de tipo de pergunta (usando delegação)
        container.addEventListener('change', (e) => {
            if (e.target.classList.contains('question-type')) {
                const questionBlock = e.target.closest('.question-block');
                if (questionBlock) {
                    this.changeQuestionType(questionBlock, e.target.value);
                }
            }
        });
        
        // Event listeners para sincronização automática quando opções perdem foco
        container.addEventListener('blur', (e) => {
            if (e.target.classList.contains('option-text')) {
                const questionBlock = e.target.closest('.question-block');
                if (questionBlock) {
                    const questionId = questionBlock.dataset.questionId;
                    this.syncQuestionDataFromDOMToRegistry(questionId);
                }
            }
        }, true); // Usar capture para garantir que seja executado
        
        // Event listeners para sincronização automática quando enunciado perde foco
        container.addEventListener('blur', (e) => {
            if (e.target.classList.contains('statement-input')) {
                const questionBlock = e.target.closest('.question-block');
                if (questionBlock) {
                    const questionId = questionBlock.dataset.questionId;
                    this.syncQuestionDataFromDOMToRegistry(questionId);
                }
            }
        }, true); // Usar capture para garantir que seja executado
        
        // Event listeners para campos de vídeo
        container.addEventListener('input', (e) => {
            if (e.target.classList.contains('video-url')) {
                const questionBlock = e.target.closest('.question-block');
                if (questionBlock) {
                    const questionId = questionBlock.dataset.questionId;
                    const url = e.target.value;
                    
                    // Atualizar preview automaticamente
                    this.updateVideoPreview(questionId, url);
                    
                    // Atualizar registry
                    const registryQuestion = this.questionsRegistry.find(q => q.id === questionId);
                    if (registryQuestion) {
                        registryQuestion.videoUrl = url;
                    }
                }
            }
        });
        
        // Event listeners para descrição do vídeo
        container.addEventListener('input', (e) => {
            if (e.target.classList.contains('video-description')) {
                const questionBlock = e.target.closest('.question-block');
                if (questionBlock) {
                    const questionId = questionBlock.dataset.questionId;
                    const description = e.target.value;
                    
                    // Atualizar registry
                    const registryQuestion = this.questionsRegistry.find(q => q.id === questionId);
                    if (registryQuestion) {
                        registryQuestion.description = description;
                    }
                }
            }
        });

        // Event listeners para campos de imagem
        container.addEventListener('input', (e) => {
            if (e.target.classList.contains('image-url')) {
                const questionBlock = e.target.closest('.question-block');
                if (questionBlock) {
                    const questionId = questionBlock.dataset.questionId;
                    const url = e.target.value;
                    
                    // Atualizar preview automaticamente
                    this.updateImagePreview(questionId, url);
                    
                    // Atualizar registry
                    const registryQuestion = this.questionsRegistry.find(q => q.id === questionId);
                    if (registryQuestion) {
                        registryQuestion.imageUrl = url;
                    }
                }
            }
        });
        
        // Event listeners para descrição da imagem
        container.addEventListener('input', (e) => {
            if (e.target.classList.contains('image-description')) {
                const questionBlock = e.target.closest('.question-block');
                if (questionBlock) {
                    const questionId = questionBlock.dataset.questionId;
                    const description = e.target.value;
                    
                    // Atualizar registry
                    const registryQuestion = this.questionsRegistry.find(q => q.id === questionId);
                    if (registryQuestion) {
                        registryQuestion.description = description;
                    }
                }
            }
        });
        
        console.log('Blocos de pergunta configurados');
    }
    
    /**
     * Sincroniza uma questão específica do DOM para o registry
     * @param {string} questionId - ID da questão
     */
    syncQuestionDataFromDOMToRegistry(questionId) {
        if (!this.questionsRegistry || !this.currentFormId) {
            console.log('Nenhum registry ativo para sincronizar');
            return;
        }

        const questionBlock = document.querySelector(`[data-question-id="${questionId}"]`);
        if (!questionBlock) {
            console.warn(`Questao ${questionId} nao encontrada no DOM`);
            return;
        }

        const registryQuestion = this.questionsRegistry.find(q => q.id === questionId);
        if (!registryQuestion) {
            console.warn(`Questao ${questionId} nao encontrada no registry`);
            return;
        }

        // Sincronizar dados específicos
        this.syncQuestionDataFromDOM(questionBlock, registryQuestion);

        // Atualizar no formRegistry principal também
        const currentFormRegistry = this.getCurrentFormRegistry();
        if (currentFormRegistry) {
            const formQuestion = currentFormRegistry.questionsRegistry.find(q => q.id === questionId);
            if (formQuestion) {
                Object.assign(formQuestion, registryQuestion);
            }
        }

    }
    
    addNewQuestion() {
        this.createMultipleChoiceQuestion('fab');
    }
    
    /**
     * Renderiza o formulário em modo preview (visualização do aluno)
     * Lê dados do questionsRegistry e renderiza sem bordas/botões de edição
     */
    renderFormPreview() {
        console.log('Renderizando preview do formulário...');
        
        const previewContainer = document.getElementById('preview-questions');
        
        if (!previewContainer) {
            console.error('Container do preview não encontrado');
            return;
        }
        
        // Verificar se há questões para renderizar
        if (!this.questionsRegistry || this.questionsRegistry.length === 0) {
            previewContainer.innerHTML = `
                <div class="preview-empty-state">
                    <div class="empty-icon">
                        <i class="material-icons">quiz</i>
                    </div>
                    <h3>Nenhuma questão criada ainda</h3>
                    <p>Adicione questões na aba "Perguntas" para vê-las aqui</p>
                </div>
            `;
            return;
        }
        
        // Obter título e descrição do formulário
        const titleElement = document.querySelector('.form-title');
        const descriptionElement = document.querySelector('.form-description-display');
        
        const formTitle = titleElement ? titleElement.textContent : 'Formulário';
        const formDescription = descriptionElement ? descriptionElement.textContent : '';
        
        // Renderizar questões ordenadas
        const sortedQuestions = [...this.questionsRegistry].sort((a, b) => a.order - b.order);
        
        let questionsHTML = '';
        sortedQuestions.forEach((question, index) => {
            questionsHTML += this.renderPreviewQuestion(question, index + 1);
        });
        
        // Criar estrutura HTML limpa apenas para respondente
        const cleanPreviewHTML = `
            <div class="preview-form">
                ${questionsHTML}
                ${this.renderPreviewSubmitButton()}
            </div>
        `;
        
        previewContainer.innerHTML = cleanPreviewHTML;
        
        // Configurar event listeners para o preview
        this.setupPreviewEventListeners();
        
        console.log('Preview renderizado com sucesso');
    }
    
    /**
     * Esconde elementos de edição quando no modo preview
     */
    hideEditingElements() {
        const elementsToHide = [
            '.form-header-editable',
            '.floating-action-bar',
            '.questions-container'
        ];
        
        elementsToHide.forEach(selector => {
            const element = document.querySelector(selector);
            if (element) {
                element.style.display = 'none';
            }
        });
        
        console.log('Elementos de edição escondidos');
    }
    
    /**
     * Mostra elementos de edição quando sai do modo preview
     */
    showEditingElements() {
        const elementsToShow = [
            '.form-header-editable',
            '.floating-action-bar',
            '.questions-container'
        ];
        
        elementsToShow.forEach(selector => {
            const element = document.querySelector(selector);
            if (element) {
                element.style.display = '';
            }
        });
        
        console.log('Elementos de edição mostrados');
    }
    
    /**
     * Renderiza uma questão individual em modo preview
     * @param {Object} question - Dados da questão do registry
     * @param {number} questionNumber - Número da questão
     * @returns {string} HTML da questão em modo preview
     */
    renderPreviewQuestion(question, questionNumber) {
        const baseHTML = `
            <div class="preview-question" data-question-id="${question.id}" data-type="${question.type}">
                <div class="preview-question-header">
                    <span class="preview-question-number">${questionNumber}.</span>
                    <span class="preview-question-statement">${this.renderMarkdown(question.enunciado || 'Enunciado da questão')}</span>
                </div>
                <div class="preview-question-content">
                    ${this.renderPreviewQuestionContent(question)}
                </div>
                <div class="preview-question-actions">
                    <button class="preview-clear-btn" data-action="clear" data-question-id="${question.id}">
                        Limpar resposta
                    </button>
                </div>
            </div>
        `;
        
        return baseHTML;
    }
    
    /**
     * Renderiza o conteúdo específico de cada tipo de questão
     * @param {Object} question - Dados da questão
     * @returns {string} HTML do conteúdo da questão
     */
    renderPreviewQuestionContent(question) {
        switch (question.type) {
            case 'multiple-choice':
                return this.renderPreviewMultipleChoice(question);
            case 'checkboxes':
                return this.renderPreviewCheckboxes(question);
            case 'short-answer':
                return this.renderPreviewShortAnswer(question);
            case 'paragraph':
                return this.renderPreviewParagraph(question);
            case 'long-answer':
                return this.renderPreviewLongAnswer(question);
            case 'file-upload':
                return this.renderPreviewFileUpload(question);
            case 'section':
                return this.renderPreviewSection(question);
            case 'video':
                return this.renderPreviewVideo(question);
            default:
                return '<p>Tipo de questão não suportado</p>';
        }
    }
    
    /**
     * Renderiza questão de múltipla escolha
     */
    renderPreviewMultipleChoice(question) {
        if (!question.itensRegistry || question.itensRegistry.length === 0) {
            return '<p>Nenhuma opção configurada</p>';
        }
        
        const optionsHTML = question.itensRegistry
            .filter(item => !item.isOtherItem)
            .sort((a, b) => a.order - b.order)
            .map(item => `
                <div class="preview-option">
                    <input type="radio" name="question_${question.id}" value="${item.id}" id="option_${question.id}_${item.id}">
                    <label for="option_${question.id}_${item.id}" class="preview-option-text">${item.text}</label>
                </div>
            `).join('');
            
        return `<div class="preview-options">${optionsHTML}</div>`;
    }
    
    /**
     * Renderiza questão de checkboxes
     */
    renderPreviewCheckboxes(question) {
        if (!question.itensRegistry || question.itensRegistry.length === 0) {
            return '<p>Nenhuma opção configurada</p>';
        }
        
        const optionsHTML = question.itensRegistry
            .filter(item => !item.isOtherItem)
            .sort((a, b) => a.order - b.order)
            .map(item => `
                <div class="preview-option">
                    <input type="checkbox" name="question_${question.id}" value="${item.id}" id="checkbox_${question.id}_${item.id}">
                    <label for="checkbox_${question.id}_${item.id}" class="preview-option-text">${item.text}</label>
                </div>
            `).join('');
            
        return `<div class="preview-options">${optionsHTML}</div>`;
    }
    
    /**
     * Renderiza questão de resposta curta
     */
    renderPreviewShortAnswer(question) {
        return `<input type="text" class="preview-text-input" placeholder="Digite sua resposta aqui..." data-question-id="${question.id}">`;
    }
    
    /**
     * Renderiza questão de parágrafo
     */
    renderPreviewParagraph(question) {
        return `<textarea class="preview-text-input preview-textarea" placeholder="Digite sua resposta aqui..." data-question-id="${question.id}"></textarea>`;
    }
    
    /**
     * Renderiza questão de resposta longa
     */
    renderPreviewLongAnswer(question) {
        return `
            <div class="preview-long-answer">
                <textarea class="preview-text-input preview-textarea" placeholder="Digite sua resposta aqui... (suporte a LaTeX e código será implementado)" data-question-id="${question.id}" rows="6"></textarea>
                <div class="preview-format-note">
                    <small>Esta questão suporta LaTeX e código (funcionalidade em desenvolvimento)</small>
                </div>
            </div>
        `;
    }
    
    /**
     * Renderiza questão de upload de arquivo
     */
    renderPreviewFileUpload(question) {
        return `
            <div class="preview-file-input" data-question-id="${question.id}">
                <input type="file" id="file_${question.id}" style="display: none;">
                <label for="file_${question.id}">
                    <i class="material-icons">cloud_upload</i>
                    <p>Clique para selecionar arquivo ou arraste aqui</p>
                    <small>Tipos suportados: todos os formatos</small>
                </label>
            </div>
        `;
    }
    
    /**
     * Renderiza seção
     */
    renderPreviewSection(question) {
        return `
            <div class="preview-section">
                <h2 class="preview-section-title">${question.enunciado || 'Título da Seção'}</h2>
            </div>
        `;
    }
    
    /**
     * Renderiza vídeo
     */
    renderPreviewVideo(question) {
        const description = question.description || '';
        const videoUrl = question.videoUrl || '';
        
        let videoHTML = '';
        if (videoUrl) {
            const videoId = this.extractYouTubeId(videoUrl);
            if (videoId) {
                videoHTML = `
                    <div class="preview-video-embed">
                        <iframe src="https://www.youtube.com/embed/${videoId}" 
                                frameborder="0" 
                                allowfullscreen>
                        </iframe>
                    </div>
                `;
            }
        }
        
        return `
            <div class="preview-video">
                ${description ? `<div class="preview-video-description">${description}</div>` : ''}
                ${videoHTML || '<p>Nenhum vídeo configurado</p>'}
            </div>
        `;
    }
    
    /**
     * Renderiza botão de envio do formulário
     */
    renderPreviewSubmitButton() {
        return `
            <div class="preview-submit-container">
                <button class="preview-submit-btn" id="preview-submit">
                    <i class="material-icons">send</i>
                    Enviar Formulário
                </button>
            </div>
        `;
    }
    
    /**
     * Retorna o título do tipo de questão
     */
    getQuestionTypeTitle(type) {
        const titles = {
            'multiple-choice': 'Múltipla escolha',
            'checkboxes': 'Caixas de seleção',
            'short-answer': 'Resposta curta',
            'paragraph': 'Parágrafo',
            'long-answer': 'Resposta longa',
            'file-upload': 'Upload de arquivo',
            'section': 'Seção',
            'video': 'Vídeo'
        };
        return titles[type] || 'Questão';
    }
    
    /**
     * Configura event listeners para o modo preview
     */
    setupPreviewEventListeners() {
        const previewContainer = document.getElementById('preview-questions');
        if (!previewContainer) return;
        
        // Event listeners para botões de limpar resposta
        previewContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('preview-clear-btn')) {
                const questionId = e.target.dataset.questionId;
                this.clearPreviewQuestionResponse(questionId);
            }
        });
        
        // Event listeners para envio do formulário
        const submitBtn = document.getElementById('preview-submit');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => {
                this.handlePreviewSubmit();
            });
        }
        
        // Event listeners para seleção de opções (visual feedback)
        previewContainer.addEventListener('change', (e) => {
            if (e.target.type === 'radio' || e.target.type === 'checkbox') {
                this.updatePreviewOptionSelection(e.target);
            }
        });
        
        // Event listeners para upload de arquivo
        previewContainer.addEventListener('change', (e) => {
            if (e.target.type === 'file') {
                this.handlePreviewFileUpload(e.target);
            }
        });
        
        console.log('Event listeners do preview configurados');
    }
    
    /**
     * Limpa a resposta de uma questão específica
     * @param {string} questionId - ID da questão
     */
    clearPreviewQuestionResponse(questionId) {
        const questionElement = document.querySelector(`[data-question-id="${questionId}"]`);
        if (!questionElement) return;
        
        const questionType = questionElement.dataset.type;
        
        switch (questionType) {
            case 'multiple-choice':
                // Limpar radio buttons
                questionElement.querySelectorAll('input[type="radio"]').forEach(radio => {
                    radio.checked = false;
                    radio.closest('.preview-option').classList.remove('selected');
                });
                break;
                
            case 'checkboxes':
                // Limpar checkboxes
                questionElement.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                    checkbox.checked = false;
                    checkbox.closest('.preview-option').classList.remove('selected');
                });
                break;
                
            case 'short-answer':
            case 'paragraph':
            case 'long-answer':
                // Limpar campos de texto
                questionElement.querySelectorAll('.preview-text-input').forEach(input => {
                    input.value = '';
                });
                break;
                
            case 'file-upload':
                // Limpar arquivo
                questionElement.querySelectorAll('input[type="file"]').forEach(input => {
                    input.value = '';
                });
                break;
        }
        
        console.log(`Resposta da questão ${questionId} limpa`);
    }
    
    /**
     * Atualiza feedback visual da seleção de opções
     * @param {HTMLElement} inputElement - Elemento de input selecionado
     */
    updatePreviewOptionSelection(inputElement) {
        const option = inputElement.closest('.preview-option');
        if (!option) return;
        
        if (inputElement.type === 'radio') {
            // Para radio buttons, remover seleção de outras opções
            const questionElement = inputElement.closest('.preview-question');
            questionElement.querySelectorAll('.preview-option').forEach(opt => {
                opt.classList.remove('selected');
            });
        }
        
        // Atualizar estado da opção atual
        if (inputElement.checked) {
            option.classList.add('selected');
        } else {
            option.classList.remove('selected');
        }
    }
    
    /**
     * Manipula upload de arquivo
     * @param {HTMLElement} fileInput - Input de arquivo
     */
    handlePreviewFileUpload(fileInput) {
        const files = fileInput.files;
        if (files.length > 0) {
            const fileName = files[0].name;
            const label = fileInput.nextElementSibling;
            if (label) {
                label.innerHTML = `
                    <i class="material-icons">check_circle</i>
                    <p>Arquivo selecionado: ${fileName}</p>
                    <small>Clique para alterar o arquivo</small>
                `;
            }
        }
    }
    
    /**
     * Coleta todas as respostas do formulário
     * @returns {Object} Objeto com as respostas organizadas por questão
     */
    collectPreviewResponses() {
        const responses = {};
        const previewContainer = document.getElementById('preview-questions');
        if (!previewContainer) return responses;
        
        const questions = previewContainer.querySelectorAll('.preview-question');
        
        questions.forEach(question => {
            const questionId = question.dataset.questionId;
            const questionType = question.dataset.type;
            
            if (!questionId || questionType === 'section' || questionType === 'video') {
                return; // Pular seções e vídeos
            }
            
            let response = null;
            
            switch (questionType) {
                case 'multiple-choice':
                    const selectedRadio = question.querySelector('input[type="radio"]:checked');
                    response = selectedRadio ? selectedRadio.value : null;
                    break;
                    
                case 'checkboxes':
                    const selectedCheckboxes = question.querySelectorAll('input[type="checkbox"]:checked');
                    response = Array.from(selectedCheckboxes).map(cb => cb.value);
                    break;
                    
                case 'short-answer':
                case 'paragraph':
                case 'long-answer':
                    const textInput = question.querySelector('.preview-text-input');
                    response = textInput ? textInput.value.trim() : '';
                    break;
                    
                case 'file-upload':
                    const fileInput = question.querySelector('input[type="file"]');
                    response = fileInput && fileInput.files.length > 0 ? fileInput.files[0].name : null;
                    break;
            }
            
            responses[questionId] = {
                type: questionType,
                response: response,
                isEmpty: this.isResponseEmpty(response)
            };
        });
        
        return responses;
    }
    
    /**
     * Verifica se uma resposta está vazia
     * @param {*} response - Resposta a verificar
     * @returns {boolean} True se a resposta estiver vazia
     */
    isResponseEmpty(response) {
        if (response === null || response === undefined) return true;
        if (typeof response === 'string') return response.trim() === '';
        if (Array.isArray(response)) return response.length === 0;
        return false;
    }
    
    /**
     * Manipula o envio do formulário
     */
    handlePreviewSubmit() {
        console.log('Enviando formulário...');
        
        // Coletar todas as respostas
        const responses = this.collectPreviewResponses();
        
        // Validar se há pelo menos uma resposta
        const hasResponses = Object.values(responses).some(r => !r.isEmpty);
        
        if (!hasResponses) {
            alert('Por favor, responda pelo menos uma questão antes de enviar o formulário.');
            return;
        }
        
        // Simular envio (por enquanto apenas mostrar no console)
        console.log('Respostas coletadas:', responses);
        
        // Mostrar confirmação
        this.showPreviewSubmitConfirmation(responses);
    }
    
    /**
     * Mostra confirmação do envio com resumo das respostas
     * @param {Object} responses - Respostas coletadas
     */
    showPreviewSubmitConfirmation(responses) {
        const totalQuestions = Object.keys(responses).length;
        const answeredQuestions = Object.values(responses).filter(r => !r.isEmpty).length;
        
        const message = `
Formulário enviado com sucesso!

Resumo:
• Total de questões: ${totalQuestions}
• Questões respondidas: ${answeredQuestions}
• Taxa de resposta: ${Math.round((answeredQuestions / totalQuestions) * 100)}%

Esta é uma visualização. Em produção, as respostas seriam enviadas para processamento.
        `;
        
        alert(message);
    }
    
    /**
     * Gera título da questão baseado no número e tipo
     * @param {number} questionNumber - Número da questão no formulário
     * @param {string} questionType - Tipo da questão
     * @returns {string} Título formatado
     */
    generateQuestionTitle(questionNumber, questionType) {
        const typeNames = {
            'multiple-choice': 'Múltipla escolha',
            'checkboxes': 'Caixas de seleção',
            'short-answer': 'Resposta curta',
            'paragraph': 'Resposta longa - Texto',
            'long-answer': 'Resposta longa - Matemática e código',
            'file-upload': 'Upload de arquivo',
            'section': 'Seção',
            'video': 'Vídeo',
            'image': 'Imagem'
        };
        
        const typeName = typeNames[questionType] || 'Questão';
        return `${questionNumber}. ${typeName}`;
    }
    
    /**
     * Atualiza título da questão no DOM baseado no registry
     * @param {HTMLElement} questionBlock - Elemento DOM da questão
     * @param {Object} questionInRegistry - Dados da questão no registry
     */
    updateQuestionTitle(questionBlock, questionInRegistry) {
        const questionTitleInput = questionBlock.querySelector('.question-title');
        if (questionTitleInput && questionInRegistry) {
            const newTitle = this.generateQuestionTitle(questionInRegistry.order, questionInRegistry.type);
            questionTitleInput.value = newTitle;
            console.log(`Titulo da questao ${questionInRegistry.id} atualizado para: ${newTitle}`);
        }
    }
    
    /**
     * Múltipla Escolha: Gera HTML para uma única opção
     * @param {Object} option - Dados da opção (id, text, isCorrect)
     * @param {number} index - Índice da opção (para numeração visual)
     * @param {string} questionType - Tipo da questão (multiple-choice, checkboxes, etc.)
     * @returns {string} HTML da opção
     */
    generateOptionHTML(option, index, questionType = 'multiple-choice') {
        const optionNumber = index + 1;
        const isOtherItem = option.text === 'outro item';
        
        // Diferentes layouts baseados no tipo de questão
        if (questionType === 'checkboxes') {
            return `
                <div class="option-item ${option.isCorrect ? 'correct-option' : ''}" 
                     data-option-id="${option.id}" 
                     data-is-other="${isOtherItem}">
                    <div class="option-preview">
                        <input type="checkbox" 
                               id="preview-${option.id}" 
                               ${option.isCorrect ? 'checked' : ''} 
                               disabled>
                        <label for="preview-${option.id}" class="option-preview-text">${option.text}</label>
                    </div>
                    <input type="text" 
                           class="option-text hidden" 
                           placeholder="Opção ${optionNumber}" 
                           value="${option.text}"
                           data-original-value="${option.text}">
                    <button class="check-option" title="Marcar como resposta correta">✓</button>
                    <button class="remove-option" title="Remover opção">×</button>
                </div>
            `;
        }
        
        // Layout padrão para múltipla escolha
        return `
            <div class="option-item ${option.isCorrect ? 'correct-option' : ''}" 
                 data-option-id="${option.id}" 
                 data-is-other="${isOtherItem}">
                <input type="text" 
                       class="option-text" 
                       placeholder="Opção ${optionNumber}" 
                       value="${option.text}"
                       data-original-value="${option.text}">
                <button class="check-option" title="Marcar como resposta correta">✓</button>
                <button class="remove-option" title="Remover opção">×</button>
            </div>
        `;
    }

    /**
     * Múltipla Escolha: Converte "outro item" em opção real e cria novo "outro item"
     * @param {HTMLElement} optionElement - Elemento da opção que foi clicada
     * @param {Object} question - Dados da questão atual
     */
    handleOtherItemClick(optionElement, question) {
        const optionId = optionElement.dataset.optionId;
        const optionIndex = question.itensRegistry.findIndex(opt => opt.id === optionId);
        
        if (optionIndex === -1) return;
        
        // Converter "outro item" para opção numerada
        const newOptionNumber = this.getNextOptionNumber(question.itensRegistry);
        question.itensRegistry[optionIndex].text = `opcao ${newOptionNumber}`;
        question.itensRegistry[optionIndex].isOtherItem = false;
        
        // Adicionar novo "outro item" no final
        const nextId = this.getNextOptionId(question.itensRegistry);
        const nextOrder = question.itensRegistry.length + 1;
        
        question.itensRegistry.push({
            id: nextId,
            text: 'outro item',
            isCorrect: false,
            position: question.itensRegistry.length,
            order: nextOrder,
            isOtherItem: true
        });
        
        // Re-renderizar opções e focar na opção que foi convertida
        this.renderQuestionOptions(question);
        this.focusOnOption(optionId);
        
        console.log(`Opcao convertida para "opcao ${newOptionNumber}"`);
    }

    /**
     * Múltipla Escolha: Obtém o próximo número de opção disponível
     * @param {Array} itensRegistry - Array de itens existentes
     * @returns {number} Próximo número de opção
     */
    getNextOptionNumber(itensRegistry) {
        const existingNumbers = itensRegistry
            .map(opt => opt.text.match(/^opcao (\d+)$/))
            .filter(match => match)
            .map(match => parseInt(match[1]));
        
        return existingNumbers.length > 0 ? Math.max(...existingNumbers) + 1 : 1;
    }

    /**
     * Múltipla Escolha: Obtém o próximo ID de opção disponível
     * @param {Array} itensRegistry - Array de itens existentes
     * @returns {string} Próximo ID de opção (a, b, c, ...)
     */
    getNextOptionId(itensRegistry) {
        const usedIds = itensRegistry.map(opt => opt.id);
        const alphabet = 'abcdefghijklmnopqrstuvwxyz';
        
        for (let char of alphabet) {
            if (!usedIds.includes(char)) {
                return char;
            }
        }
        
        // Fallback para IDs longos se necessário
        return `opt_${Date.now()}`;
    }

    /**
     * Múltipla Escolha: Renderiza todas as opções de uma questão
     * @param {Object} question - Dados da questão
     */
    renderQuestionOptions(question) {
        if (!question.itensRegistry) return;
        
        const questionBlock = document.querySelector(`[data-question-id="${question.id}"]`);
        if (!questionBlock) return;
        
        const optionsContainer = questionBlock.querySelector('.question-options');
        if (!optionsContainer) return;
        
        // Usar tipo da questão do registry (fonte única de verdade) 
        let questionType = question.type;
        
        // Fallback para o valor do select se o tipo não estiver definido no objeto
        if (!questionType) {
            const questionTypeSelect = questionBlock.querySelector('.question-type');
            questionType = questionTypeSelect ? questionTypeSelect.value : 'multiple-choice';
        }
        
        const optionsHTML = question.itensRegistry
            .map((option, index) => this.generateOptionHTML(option, index, questionType))
            .join('');
        
        optionsContainer.innerHTML = optionsHTML;
    }

    /**
     * Múltipla Escolha: Define foco em uma opção específica
     * @param {string} optionId - ID da opção para focar
     */
    focusOnOption(optionId) {
        setTimeout(() => {
            const optionInput = document.querySelector(`[data-option-id="${optionId}"] .option-text`);
            if (optionInput) {
                optionInput.focus();
                optionInput.select(); // Seleciona todo o texto para facilitar edição
            }
        }, 100); // Pequeno delay para garantir que o DOM foi atualizado
    }

    

    /**
     * Sincroniza mudanças do DOM de volta para o registry
     * @param {string} questionId - ID da questão a ser sincronizada
     * @param {HTMLElement} questionBlock - Elemento DOM da questão (opcional)
     */
    syncDOMToRegistry(questionId, questionBlock = null) {
        if (!questionBlock) {
            questionBlock = document.querySelector(`[data-question-id="${questionId}"]`);
        }
        
        if (!questionBlock) {
            console.warn(`Não foi possível encontrar questionBlock para ID: ${questionId}`);
            return;
        }

        const registryQuestion = this.getQuestionFromRegistry(questionId);
        if (!registryQuestion) {
            console.warn(`Questão não encontrada no registry: ${questionId}`);
            return;
        }

        // Atualizar título se mudou
        const titleInput = questionBlock.querySelector('.question-title');
        if (titleInput && titleInput.value !== registryQuestion.title) {
            registryQuestion.title = titleInput.value;
        }

        // Atualizar estado de seleção baseado na classe CSS
        const isSelected = questionBlock.classList.contains('selected');
        if (isSelected !== registryQuestion.selecionado) {
            registryQuestion.selecionado = isSelected;
            registryQuestion.focus = isSelected;
        }

        console.log(`Registry sincronizado para questao ${questionId}`);
    }

    /**
     * Debug: Exibe estado completo do questionsRegistry no console
     * Útil para depuração e verificação do estado das questões
     */
    debugQuestionsRegistry() {
        console.group('Estado do Questions Registry');
        
        // Informações do formulário
        const formInfo = this.getCurrentFormInfo();
        console.log(`ID do formulario: ${formInfo.formId || 'Nao inicializado'}`);
        console.log(`Total de questoes: ${formInfo.questionCount}`);
        console.log(`Tem questoes: ${formInfo.hasQuestions ? 'sim' : 'nao'}`);
        console.log(`Registry valido: ${this.questionsRegistry !== null ? 'sim' : 'nao'}`);
        console.log('');
        
        if (!this.questionsRegistry) {
            console.warn('Registry nao inicializado para este formulario');
            console.groupEnd();
            return null;
        }
        
        if (this.questionsRegistry.length === 0) {
            console.log('Nenhuma questao criada ainda');
            console.groupEnd();
            return this.questionsRegistry;
        }
        
        this.questionsRegistry.forEach((question, index) => {
            console.group(`Questao ${index + 1} (ID: ${question.id})`);
            console.log(`Posicao: ${question.position}`);
            console.log(`Ordem: ${question.order}`);
            console.log(`Foco: ${question.focus ? 'sim' : 'nao'}`);
            console.log(`Selecionado: ${question.selecionado ? 'sim' : 'nao'}`);
            console.log(`Rubrica Form: ${question.rubricaform ? 'sim' : 'nao'}`);
            console.log(`Enunciado: "${question.enunciado}"`);
            console.log(`Justificativa: "${question.justificativa}"`);
            console.log(`Rubrica: "${question.rubrica}"`);
            console.groupEnd();
        });
        
        console.groupEnd();
        return this.questionsRegistry;
    }

    /**
     * Edita o enunciado de uma questão (transforma display em textarea)
     * @param {string} questionId - ID da questão
     */
    editQuestionStatement(questionId) {
        const questionBlock = document.querySelector(`[data-question-id="${questionId}"]`);
        if (!questionBlock) return;

        const statementDisplay = questionBlock.querySelector('.statement-display');
        const statementInput = questionBlock.querySelector('.statement-input');
        
        if (!statementDisplay || !statementInput) return;

        // Pegar enunciado atual do registry
        const registryData = this.getQuestionFromRegistry(questionId);
        const currentStatement = registryData ? registryData.enunciado : '';
        
        // Transferir texto para o textarea
        statementInput.value = currentStatement;
        
        // Configurar altura inicial (mesma do justification-input)
        statementInput.style.minHeight = '60px';
        statementInput.style.height = '60px';
        statementInput.style.resize = 'vertical';
        
        // Alternar visibilidade
        statementDisplay.classList.add('hidden');
        statementInput.classList.remove('hidden');
        
        // Focar e selecionar o texto
        statementInput.focus();
        statementInput.select();
        
        console.log(`Editando enunciado da questao ${questionId}`);
    }

    /**
     * Salva o enunciado da questão (onblur do textarea)
     * @param {string} questionId - ID da questão
     */
    saveQuestionStatement(questionId) {
        const questionBlock = document.querySelector(`[data-question-id="${questionId}"]`);
        if (!questionBlock) return;

        const statementDisplay = questionBlock.querySelector('.statement-display');
        const statementInput = questionBlock.querySelector('.statement-input');
        
        if (!statementDisplay || !statementInput) return;

        const newStatement = statementInput.value.trim();
        
        // Salvar no registry
        this.setQuestionEnunciado(questionId, newStatement);
        
        // Sincronizar com formRegistry
        const currentFormRegistry = this.getCurrentFormRegistry();
        if (currentFormRegistry) {
            currentFormRegistry.questionsRegistry = this.questionsRegistry;
            currentFormRegistry.modified = new Date();
        }
        
        // Atualizar display com renderização markdown
        if (newStatement) {
            const renderedHTML = this.renderMarkdown(newStatement);
            statementDisplay.innerHTML = renderedHTML;
        } else {
            statementDisplay.textContent = 'Clique para adicionar o enunciado da questão';
        }
        
        // Alternar visibilidade
        statementInput.classList.add('hidden');
        statementDisplay.classList.remove('hidden');
        
        console.log(`Enunciado salvo para questao ${questionId}`);
    }

    /**
     * Lida com teclas especiais no textarea do enunciado
     * @param {Event} event - Evento de teclado
     * @param {string} questionId - ID da questão
     */
    handleStatementKeydown(event, questionId) {
        // Esc para cancelar edição
        if (event.key === 'Escape') {
            const questionBlock = document.querySelector(`[data-question-id="${questionId}"]`);
            if (!questionBlock) return;

            const statementDisplay = questionBlock.querySelector('.statement-display');
            const statementInput = questionBlock.querySelector('.statement-input');
            
            // Restaurar texto original sem salvar
            const registryData = this.getQuestionFromRegistry(questionId);
            const originalStatement = registryData ? registryData.enunciado : '';
            statementInput.value = originalStatement;
            
            // Voltar para display
            statementInput.classList.add('hidden');
            statementDisplay.classList.remove('hidden');
            
            event.preventDefault();
        }
        
        // Enter para salvar e sair (sem permitir quebra de linha)
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.saveQuestionStatement(questionId);
        }
    }

    /**
     * Atualiza o display do enunciado baseado no registry
     * @param {string} questionId - ID da questão
     */
    updateStatementDisplay(questionId) {
        const questionBlock = document.querySelector(`[data-question-id="${questionId}"]`);
        if (!questionBlock) return;

        const statementDisplay = questionBlock.querySelector('.statement-display');
        if (!statementDisplay) return;

        // Pegar enunciado do registry
        const registryData = this.getQuestionFromRegistry(questionId);
        const enunciado = registryData ? registryData.enunciado : '';
        
        console.log(`Atualizando display questao ${questionId}, enunciado:`, enunciado);
        
        // Atualizar display com renderização markdown
        if (enunciado && enunciado.trim()) {
            if (typeof marked === 'undefined') {
                console.warn(`Marked.js nao disponivel ao renderizar questao ${questionId}, tentando novamente...`);
                // Tentar novamente após delay se Marked.js não estiver carregado
                setTimeout(() => {
                    this.updateStatementDisplay(questionId);
                }, 200);
                return;
            }
            
            const renderedHTML = this.renderMarkdown(enunciado);
            statementDisplay.innerHTML = renderedHTML;
            
            // Aplicar estilo apropriado baseado no conteúdo
            if (renderedHTML.includes('<') && renderedHTML.includes('>')) {
                // Contém HTML (markdown renderizado)
                statementDisplay.style.whiteSpace = 'normal';
            } else {
                // Texto simples
                statementDisplay.style.whiteSpace = 'pre-wrap';
            }
            
            console.log(`Markdown renderizado para questao ${questionId}`, {
                innerHTML: statementDisplay.innerHTML,
                textContent: statementDisplay.textContent
            });
        } else {
            statementDisplay.textContent = 'Clique para adicionar o enunciado da questão';
            console.log(`Enunciado vazio para questao ${questionId}`);
        }
    }

    /**
     * Extrai ID do YouTube de uma URL
     * @param {string} url - URL do YouTube
     * @returns {string|null} ID do vídeo ou null se inválido
     */
    extractYouTubeId(url) {
        if (!url) return null;
        
        // Padrões de URL do YouTube
        const patterns = [
            /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
            /youtube\.com\/v\/([^&\n?#]+)/
        ];
        
        for (let pattern of patterns) {
            const match = url.match(pattern);
            if (match && match[1]) {
                return match[1];
            }
        }
        
        return null;
    }

    /**
     * Gera preview do vídeo do YouTube
     * @param {string} videoId - ID do vídeo do YouTube
     * @returns {string} HTML do iframe
     */
    generateYouTubePreview(videoId) {
        return `
            <iframe 
                src="https://www.youtube.com/embed/${videoId}" 
                title="YouTube video player" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
        `;
    }

    /**
     * Atualiza preview do vídeo baseado na URL
     * @param {string} questionId - ID da questão/vídeo
     * @param {string} url - URL do vídeo
     */
    updateVideoPreview(questionId, url) {
        const previewContainer = document.getElementById(`video-preview-${questionId}`);
        if (!previewContainer) return;
        
        if (!url || !url.trim()) {
            // URL vazia - mostrar placeholder normal
            previewContainer.className = 'video-preview placeholder';
            previewContainer.innerHTML = `
                <div class="preview-placeholder">
                    <i class="material-icons">videocam</i>
                    <span>Preview do vídeo aparecerá aqui após inserir o link</span>
                </div>
            `;
            console.log(`URL vazia para vídeo ${questionId}`);
            return;
        }
        
        const videoId = this.extractYouTubeId(url);
        
        if (videoId) {
            // URL válida - mostrar vídeo
            previewContainer.className = 'video-preview has-video';
            previewContainer.innerHTML = this.generateYouTubePreview(videoId);
            console.log(`Preview gerado para vídeo ${questionId}:`, videoId);
        } else {
            // URL inválida - mostrar erro
            previewContainer.className = 'video-preview placeholder error';
            previewContainer.innerHTML = `
                <div class="preview-placeholder">
                    <i class="material-icons error-icon">error</i>
                    <span>URL inválida. Use um link válido do YouTube</span>
                </div>
            `;
            console.log(`URL inválida para vídeo ${questionId}:`, url);
        }
    }

    updateImagePreview(questionId, url) {
        const previewContainer = document.getElementById(`image-preview-${questionId}`);
        if (!previewContainer) return;
        
        if (!url || !url.trim()) {
            // URL vazia - mostrar placeholder normal
            previewContainer.className = 'image-preview placeholder';
            previewContainer.innerHTML = `
                <div class="preview-placeholder">
                    <i class="material-icons">image</i>
                    <span>Preview da imagem aparecerá aqui após inserir uma URL válida</span>
                </div>
            `;
            console.log(`URL vazia para imagem ${questionId}`);
        } else if (this.isValidImageUrl(url)) {
            // URL válida - mostrar imagem
            previewContainer.className = 'image-preview has-image';
            previewContainer.innerHTML = `<img src="${url}" alt="Preview da imagem" onerror="this.parentElement.className='image-preview placeholder error'; this.parentElement.innerHTML='<div class=\\'preview-placeholder\\'><i class=\\'material-icons error-icon\\'>error</i><span>Erro ao carregar a imagem</span></div>';">`;
            console.log(`Preview gerado para imagem ${questionId}:`, url);
        } else {
            // URL inválida - mostrar erro
            previewContainer.className = 'image-preview placeholder error';
            previewContainer.innerHTML = `
                <div class="preview-placeholder">
                    <i class="material-icons error-icon">error</i>
                    <span>URL inválida. Use uma URL de imagem válida (jpg, png ou webp)</span>
                </div>
            `;
            console.log(`URL inválida para imagem ${questionId}:`, url);
        }
    }

    isValidImageUrl(url) {
        if (!url) return false;
        
        try {
            const urlObj = new URL(url);
            // Verificar se é um protocolo válido
            if (!['http:', 'https:'].includes(urlObj.protocol)) return false;
            
            // Verificar apenas extensões permitidas: jpg, png e webp
            const imageExtensions = /\.(jpg|jpeg|png|webp)$/i;
            return imageExtensions.test(urlObj.pathname) || 
                   url.includes('imgur.com') || 
                   url.includes('googleusercontent.com') ||
                   url.includes('unsplash.com') ||
                   url.includes('pexels.com');
        } catch {
            return false;
        }
    }

    /**
     * Inicializa um novo registry específico para o formulário atual
     * Chamado sempre que um novo formulário é criado/carregado
     */
    initializeFormRegistry() {
        // Criar novo formRegistry
        const formRegistry = this.createFormRegistry();
        
        // Ativar o formulário recém-criado
        this.activateForm(formRegistry.id);
        
        // Renderizar cards na sidebar
        this.renderFormCards();
        
        console.log(`Registry inicializado para formulario: ${this.currentFormId}`);
        console.log('Registry anterior limpo, iniciando com array vazio');
    }

    /**
     * Desativa o formulário atual (mas não destrói - mantém em memória)
     * Chamado ao navegar para fora do editor de formulário
     */
    destroyFormRegistry() {
        if (this.currentFormId) {
            console.log(`Desativando formulario: ${this.currentFormId}`);
            
            // Sincronizar todas as questões do DOM para o registry ANTES de desativar
            this.syncAllQuestionsToRegistry();
            
            // Atualizar metadados antes de desativar
            this.updateCurrentFormMetadata();
            
            // Desativar mas não destruir
            this.questionsRegistry = null;
            this.currentFormId = null;
            
            // Atualizar visual
            this.updateFormCardStates();
            
            console.log('Formulario desativado com sucesso');
        }
    }

    /**
     * Sincroniza todas as questões visíveis no DOM para o registry
     */
    syncAllQuestionsToRegistry() {
        if (!this.questionsRegistry || !this.currentFormId) {
            console.log('Nenhum registry ativo para sincronizar');
            return;
        }

        console.log('Sincronizando todas as questoes do DOM para o registry...');
        
        // Encontrar todas as questões no DOM
        const questionBlocks = document.querySelectorAll('.question-block[data-question-id]');
        const currentFormRegistry = this.getCurrentFormRegistry();
        
        if (!currentFormRegistry) {
            console.error('FormRegistry nao encontrado para sincronizacao');
            return;
        }
        
        // Criar um novo array para armazenar questões sincronizadas
        const syncedQuestions = [];
        
        questionBlocks.forEach((questionBlock, index) => {
            const questionId = questionBlock.dataset.questionId;
            
            let registryQuestion = this.questionsRegistry.find(q => q.id === questionId);
            
            if (!registryQuestion) {
                registryQuestion = {
                    id: questionId,
                    type: 'multiple-choice',
                    position: index,
                    order: index + 1,
                    focus: false,
                    enunciado: '',
                    justificativa: '',
                    rubrica: '',
                    selecionado: false,
                    rubricaform: false,
                    itensRegistry: [],
                    domElement: questionBlock
                };
            }
            
            this.syncQuestionDataFromDOM(questionBlock, registryQuestion);
            
            registryQuestion.position = index;
            registryQuestion.order = index + 1;
            
            syncedQuestions.push(registryQuestion);
        });
        
        // Atualizar o questionsRegistry com as questões sincronizadas
        this.questionsRegistry = syncedQuestions;
        currentFormRegistry.questionsRegistry = syncedQuestions;
        
        console.log(`${syncedQuestions.length} questoes sincronizadas`);
    }
    
    /**
     * Sincroniza dados específicos de uma questão do DOM para o registry
     * @param {HTMLElement} questionBlock - Elemento DOM da questão
     * @param {Object} registryQuestion - Objeto da questão no registry
     */
    syncQuestionDataFromDOM(questionBlock, registryQuestion) {
        // NOTA: Enunciado não é sincronizado aqui pois é gerenciado por editQuestionStatement/saveQuestionStatement
        // Usar textContent destruiria a formatação markdown original
        
        // Sincronizar opções
        const optionInputs = questionBlock.querySelectorAll('.option-text');
        const itensRegistry = [];
        
        optionInputs.forEach((input, index) => {
            const optionId = input.dataset.optionId || String.fromCharCode(97 + index); // a, b, c...
            const text = input.value.trim();
            const isCorrect = input.closest('.option-item').classList.contains('correct-option');
            const isOtherItem = text === 'outro item' || input.closest('.option-item').classList.contains('other-item');
            
            if (text) {
                itensRegistry.push({
                    id: optionId,
                    text: text,
                    isCorrect: isCorrect,
                    position: index,
                    order: index + 1,
                    isOtherItem: isOtherItem
                });
            }
        });
        
        // Garantir que sempre temos pelo menos a opção "outro item"
        if (itensRegistry.length === 0 || !itensRegistry.some(item => item.isOtherItem)) {
            itensRegistry.push({
                id: String.fromCharCode(97 + itensRegistry.length),
                text: 'outro item',
                isCorrect: false,
                position: itensRegistry.length,
                order: itensRegistry.length + 1,
                isOtherItem: true
            });
        }
        
        registryQuestion.itensRegistry = itensRegistry;
        
        console.log(`Questao ${registryQuestion.id} sincronizada`);
    }

    /**
     * Debug simples dos registries
     */
    debugRegistries() {
        console.log('CurrentFormId:', this.currentFormId);
        console.log('FormsRegistry size:', this.formsRegistry.size);
        
        if (this.questionsRegistry) {
            console.log(`QuestionsRegistry: ${this.questionsRegistry.length} questoes`);
        }
    }

    /**
     * Obtém informações do formulário atual
     * @returns {Object} Informações do formulário e registry
     */
    getCurrentFormInfo() {
        return {
            formId: this.currentFormId,
            questionCount: this.questionsRegistry ? this.questionsRegistry.length : 0,
            hasQuestions: this.questionsRegistry && this.questionsRegistry.length > 0,
            registrySize: this.questionsRegistry ? this.questionsRegistry.length : 0
        };
    }
    
    removeOption(optionItem) {
        const questionBlock = optionItem.closest('.question-block');
        const questionId = questionBlock.dataset.questionId;
        const optionId = optionItem.dataset.optionId;
        
        // Buscar questão no registry
        const question = this.getQuestionFromRegistry(questionId);
        if (!question || !question.itensRegistry) {
            console.log('Questao nao encontrada no registry para remocao de opcao');
            return;
        }
        
        // Não permitir remover se só há 2 opções
        if (question.itensRegistry.length <= 2) {
            console.log('Nao e possivel remover: minimo de 2 opcoes necessario');
            return;
        }
        
        // Remover do itensRegistry
        const optionIndex = question.itensRegistry.findIndex(opt => opt.id === optionId);
        if (optionIndex !== -1) {
            question.itensRegistry.splice(optionIndex, 1);
            
            // Atualizar posições dos itens restantes
            question.itensRegistry.forEach((item, index) => {
                item.position = index;
                item.order = index + 1;
            });
            
            // Re-renderizar opções
            this.renderQuestionOptions(question);
            
            console.log(`Opcao ${optionId} removida do bloco ${questionId}`);
        }
    }
    
    changeQuestionType(questionBlock, newType) {
        console.log(`Mudando tipo de pergunta para: ${newType}`);
        
        const questionId = questionBlock.dataset.questionId;
        const optionsContainer = questionBlock.querySelector('.question-options');
        
        // Atualizar tipo no registry
        if (this.questionsRegistry) {
            const questionInRegistry = this.questionsRegistry.find(q => q.id === questionId);
            if (questionInRegistry) {
                questionInRegistry.type = newType;
                console.log(`Tipo da questao ${questionId} atualizado no registry para: ${newType}`);
                
                // Atualizar título da questão
                this.updateQuestionTitle(questionBlock, questionInRegistry);
            }
        }
        
        // Remover área de justificativa se não for múltipla escolha
        if (newType !== 'multiple-choice' && newType !== 'checkboxes') {
            const justificationArea = questionBlock.querySelector('.justification-area');
            if (justificationArea) {
                justificationArea.remove();
                console.log('Area de justificativa removida - nao e multipla escolha');
            }
        }
        
        // Implementar diferentes layouts baseados no tipo
        switch (newType) {
            case 'multiple-choice':
            case 'checkboxes':
                // Verificar se já existem opções na questão do registry
                const questionInRegistry = this.questionsRegistry ? 
                    this.questionsRegistry.find(q => q.id === questionId) : null;
                
                if (questionInRegistry && questionInRegistry.itensRegistry && questionInRegistry.itensRegistry.length > 0) {
                    // Re-renderizar opções existentes com o novo tipo
                    questionInRegistry.type = newType; // Garantir que o tipo está atualizado
                    this.renderQuestionOptions(questionInRegistry);
                    console.log(`Opcoes re-renderizadas para tipo: ${newType}`);
                } else if (!optionsContainer.querySelector('.option-item')) {
                    // Criar questão temporária com 2 opções padrão se não existem opções
                    const tempQuestion = {
                        id: questionBlock.dataset.questionId,
                        type: newType,
                        itensRegistry: [
                            { 
                                id: 'a', 
                                text: 'opcao 1', 
                                isCorrect: false,
                                position: 0,
                                order: 1,
                                isOtherItem: false
                            },
                            { 
                                id: 'b', 
                                text: 'outro item', 
                                isCorrect: false,
                                position: 1,
                                order: 2,
                                isOtherItem: true
                            }
                        ]
                    };
                    this.renderQuestionOptions(tempQuestion);
                    console.log('Layout de multipla escolha recriado com nova UX');
                }
                break;
            case 'short-answer':
                optionsContainer.innerHTML = `
                    <div class="short-answer-preview">
                        <div class="answer-hint">
                            <i class="material-icons">info</i>
                            Os usuários terão acesso a um campo de texto curto para resposta objetiva.
                        </div>
                    </div>
                `;
                break;
            case 'paragraph':
                optionsContainer.innerHTML = `
                    <div class="paragraph-answer-preview">
                        <div class="answer-hint">
                            <i class="material-icons">info</i>
                            Os usuários terão acesso a um bloco de textos grande para a redação da resposta.
                        </div>
                    </div>
                `;
                break;
            case 'long-answer':
                optionsContainer.innerHTML = `
                    <div class="long-answer-preview">
                        <div class="answer-hint">
                            <i class="material-icons">code</i>
                            Os usuários terão acesso a editores avançados de LaTeX e código com syntax highlighting.
                        </div>
                    </div>
                `;
                break;
            case 'file-upload':
                optionsContainer.innerHTML = `
                    <div class="file-upload-preview">
                        <div class="answer-hint">
                            <i class="material-icons">upload_file</i>
                            Os usuários poderão fazer upload de arquivos como anexos à resposta.
                        </div>
                    </div>
                `;
                break;
            case 'section':
                // Para seções, precisa reconstruir todo o bloco usando a função única
                this.convertQuestionToSection(questionBlock);
                return; // Sair da função pois o bloco foi reconstruído
            case 'video':
                // Para vídeos, precisa reconstruir todo o bloco usando a função única  
                this.convertQuestionToVideo(questionBlock);
                return; // Sair da função pois o bloco foi reconstruído
            case 'image':
                // Para imagens, precisa reconstruir todo o bloco usando a função única
                this.convertQuestionToImage(questionBlock);
                return; // Sair da função pois o bloco foi reconstruído
        }
        
        console.log(`Tipo de pergunta alterado para ${newType}`);
    }

    /**
     * Converte uma questão existente para seção usando a função única
     * @param {HTMLElement} questionBlock - Bloco da questão a ser convertida
     */
    convertQuestionToSection(questionBlock) {
        const questionId = questionBlock.dataset.questionId;
        
        // Buscar questão no registry
        const questionInRegistry = this.questionsRegistry.find(q => q.id === questionId);
        if (!questionInRegistry) {
            console.warn(`Questao ${questionId} nao encontrada no registry`);
            return;
        }
        
        // Sincronizar dados antes da conversão
        this.syncQuestionDataFromDOMToRegistry(questionId);
        
        // Atualizar tipo no registry
        questionInRegistry.type = 'section';
        
        // Determinar posição atual
        const currentIndex = this.questionsRegistry.findIndex(q => q.id === questionId);
        
        // Remover questão antiga do DOM
        questionBlock.remove();
        
        // Criar dados para seção mantendo o ID e posição
        const sectionData = {
            id: questionId,
            number: questionInRegistry.order,
            type: 'section',
            title: this.generateQuestionTitle(questionInRegistry.order, 'section'),
            itensRegistry: []
        };
        
        // Criar novo HTML da seção na mesma posição
        this.createQuestionHTML(sectionData, currentIndex);
        
        console.log(`Questao ${questionId} convertida para seção`);
    }

    /**
     * Converte uma questão existente para vídeo usando a função única
     * @param {HTMLElement} questionBlock - Bloco da questão a ser convertida
     */
    convertQuestionToVideo(questionBlock) {
        const questionId = questionBlock.dataset.questionId;
        
        // Buscar questão no registry
        const questionInRegistry = this.questionsRegistry.find(q => q.id === questionId);
        if (!questionInRegistry) {
            console.warn(`Questao ${questionId} nao encontrada no registry`);
            return;
        }
        
        // Sincronizar dados antes da conversão
        this.syncQuestionDataFromDOMToRegistry(questionId);
        
        // Atualizar tipo no registry e adicionar campos de vídeo
        questionInRegistry.type = 'video';
        questionInRegistry.description = questionInRegistry.description || '';
        questionInRegistry.videoUrl = questionInRegistry.videoUrl || '';
        
        // Determinar posição atual
        const currentIndex = this.questionsRegistry.findIndex(q => q.id === questionId);
        
        // Remover questão antiga do DOM
        questionBlock.remove();
        
        // Criar dados para vídeo mantendo o ID e posição
        const videoData = {
            id: questionId,
            number: questionInRegistry.order,
            type: 'video',
            title: this.generateQuestionTitle(questionInRegistry.order, 'video'),
            description: questionInRegistry.description,
            videoUrl: questionInRegistry.videoUrl,
            itensRegistry: []
        };
        
        // Criar novo HTML do vídeo na mesma posição
        this.createQuestionHTML(videoData, currentIndex);
        
        console.log(`Questao ${questionId} convertida para vídeo`);
    }

    /**
     * Converte uma questão existente para imagem usando a função única
     * @param {HTMLElement} questionBlock - Bloco da questão a ser convertida
     */
    convertQuestionToImage(questionBlock) {
        const questionId = questionBlock.dataset.questionId;
        
        // Buscar questão no registry
        const questionInRegistry = this.questionsRegistry.find(q => q.id === questionId);
        if (!questionInRegistry) {
            console.warn(`Questao ${questionId} nao encontrada no registry`);
            return;
        }
        
        // Sincronizar dados antes da conversão
        this.syncQuestionDataFromDOMToRegistry(questionId);
        
        // Atualizar tipo no registry e adicionar campos de imagem
        questionInRegistry.type = 'image';
        questionInRegistry.description = questionInRegistry.description || '';
        questionInRegistry.imageUrl = questionInRegistry.imageUrl || '';
        
        // Determinar posição atual
        const currentIndex = this.questionsRegistry.findIndex(q => q.id === questionId);
        
        // Remover questão antiga do DOM
        questionBlock.remove();
        
        // Criar dados para imagem mantendo o ID e posição
        const imageData = {
            id: questionId,
            number: questionInRegistry.order,
            type: 'image',
            title: this.generateQuestionTitle(questionInRegistry.order, 'image'),
            description: questionInRegistry.description,
            imageUrl: questionInRegistry.imageUrl,
            itensRegistry: []
        };
        
        // Criar novo HTML da imagem na mesma posição
        this.createQuestionHTML(imageData, currentIndex);
        
        console.log(`Questao ${questionId} convertida para imagem`);
    }
    
    deleteQuestionBlock(questionBlock, questionId) {
        console.log(`Tentando excluir bloco de pergunta ${questionId}...`);
        
        // Prevenir múltiplas execuções
        if (questionBlock.dataset.deleting === 'true') {
            return;
        }
        
        // Marcar como sendo deletado
        questionBlock.dataset.deleting = 'true';
        
        if (confirm('Tem certeza que deseja excluir esta pergunta?')) {
            // Remover o bloco do DOM
            questionBlock.remove();
            
            // Renumerar as perguntas restantes
            this.renumberQuestions();
            
            // Se não há mais perguntas, mostrar estado vazio
            const container = document.querySelector('.questions-container');
            const remainingQuestions = container.querySelectorAll('.question-block');
            
            if (remainingQuestions.length === 0) {
                this.showEmptyQuestionsState();
            } else {
                // Selecionar a primeira pergunta restante se nenhuma estiver selecionada
                const selectedQuestion = container.querySelector('.question-block.selected');
                if (!selectedQuestion && remainingQuestions.length > 0) {
                    remainingQuestions[0].classList.add('selected');
                }
            }
            
            console.log(`Pergunta ${questionId} excluida com sucesso`);
        } else {
            // Se cancelou, remover a flag
            questionBlock.dataset.deleting = 'false';
        }
    }
    
    duplicateQuestionBlock(questionBlock) {
        console.log(`Duplicando bloco de pergunta...`);
        
        try {
            const questionId = questionBlock.dataset.questionId;
            
            // Buscar questão no registry
            const questionInRegistry = this.questionsRegistry.find(q => q.id === questionId);
            if (!questionInRegistry) {
                console.warn(`Questao ${questionId} nao encontrada no registry`);
                return;
            }
            
            // Sincronizar dados do DOM para o registry antes de duplicar
            this.syncQuestionDataFromDOMToRegistry(questionId);
            
            // Determinar posição de inserção (após a questão atual)
            const currentIndex = this.questionsRegistry.findIndex(q => q.id === questionId);
            const insertPosition = currentIndex + 1;
            
            // Gerar novo ID e número para a questão duplicada
            const newQuestionId = `q_${Date.now()}`;
            const newQuestionNumber = this.questionsRegistry.length + 1;
            
            // Criar cópia da questão no registry
            const duplicatedQuestion = {
                id: newQuestionId,
                type: questionInRegistry.type,
                position: insertPosition,
                order: newQuestionNumber,
                focus: true,
                enunciado: questionInRegistry.enunciado,
                justificativa: questionInRegistry.justificativa,
                rubrica: questionInRegistry.rubrica,
                selecionado: false,
                rubricaform: false,
                itensRegistry: questionInRegistry.itensRegistry ? 
                    JSON.parse(JSON.stringify(questionInRegistry.itensRegistry)) : [],
                description: questionInRegistry.description || '',
                videoUrl: questionInRegistry.videoUrl || '',
                imageUrl: questionInRegistry.imageUrl || '',
                domElement: null
            };
            
            // Remover foco de outras questões
            this.questionsRegistry.forEach(q => q.focus = false);
            
            // Adicionar ao registry
            this.questionsRegistry.splice(insertPosition, 0, duplicatedQuestion);
            
            // Atualizar posições e ordens
            this.updateQuestionsOrder();
            
            // Criar dados para o HTML
            const newQuestionData = {
                id: newQuestionId,
                number: newQuestionNumber,
                type: questionInRegistry.type,
                title: this.generateQuestionTitle(newQuestionNumber, questionInRegistry.type),
                description: questionInRegistry.description || '',
                videoUrl: questionInRegistry.videoUrl || '',
                itensRegistry: duplicatedQuestion.itensRegistry
            };
            
            // Criar HTML da questão duplicada
            this.createQuestionHTML(newQuestionData, insertPosition);
            
            // Renderizar opções se for questão com itens
            if (questionInRegistry.type === 'multiple-choice' || questionInRegistry.type === 'checkboxes') {
                setTimeout(() => {
                    this.renderQuestionOptions(duplicatedQuestion);
                    
                    // Re-renderizar justificativa compacta se existir
                    if (duplicatedQuestion.justificativa && duplicatedQuestion.justificativa.trim()) {
                        const questionBlock = document.querySelector(`[data-question-id="${newQuestionId}"]`);
                        if (questionBlock) {
                            const existingCompact = questionBlock.querySelector('.justification-compact');
                            if (!existingCompact) {
                                this.renderJustificationCompact(questionBlock, duplicatedQuestion.justificativa);
                                this.disableCorrectOptionCheck(questionBlock);
                            }
                        }
                    }
                }, 100);
            }
            
            // Restaurar dados de vídeo se for um bloco de vídeo
            if (questionInRegistry.type === 'video') {
                setTimeout(() => {
                    // Restaurar descrição
                    const descriptionField = document.getElementById(`video-description-${newQuestionId}`);
                    if (descriptionField && questionInRegistry.description) {
                        descriptionField.value = questionInRegistry.description;
                    }
                    
                    // Restaurar URL
                    const urlField = document.getElementById(`video-url-${newQuestionId}`);
                    if (urlField && questionInRegistry.videoUrl) {
                        urlField.value = questionInRegistry.videoUrl;
                        
                        // Atualizar preview automaticamente
                        this.updateVideoPreview(newQuestionId, questionInRegistry.videoUrl);
                    }
                }, 100);
            } else if (questionInRegistry.type === 'image') {
                setTimeout(() => {
                    // Restaurar dados de imagem
                    const descField = document.getElementById(`image-description-${newQuestionId}`);
                    if (descField && questionInRegistry.description) {
                        descField.value = questionInRegistry.description;
                    }
                    
                    const urlField = document.getElementById(`image-url-${newQuestionId}`);
                    if (urlField && questionInRegistry.imageUrl) {
                        urlField.value = questionInRegistry.imageUrl;
                        
                        // Atualizar preview automaticamente
                        this.updateImagePreview(newQuestionId, questionInRegistry.imageUrl);
                    }
                }, 100);
            }
            
            // Atualizar metadados do formulário
            this.updateCurrentFormMetadata();
            this.renderFormCards();
            
            console.log(`Questao duplicada com sucesso: ${newQuestionId}`);
            
        } catch (error) {
            console.error('Erro ao duplicar pergunta:', error);
            alert('Erro ao duplicar a pergunta. Tente novamente.');
        }
    }
    
    renumberQuestions() {
        const questionBlocks = document.querySelectorAll('.question-block');
        questionBlocks.forEach((block, index) => {
            const newNumber = index + 1;
            block.dataset.questionId = newNumber;
            
            // Atualizar numeração visual se houver elemento que mostra o número
            const numberElement = block.querySelector('.question-number');
            if (numberElement) {
                numberElement.textContent = newNumber;
            }
        });
        console.log(`${questionBlocks.length} questoes renumeradas`);
    }

    toggleCorrectOption(optionItem) {
        const questionBlock = optionItem.closest('.question-block');
        const questionId = questionBlock.dataset.questionId;
        const optionId = optionItem.dataset.optionId;
        
        // Buscar questão no registry
        const question = this.getQuestionFromRegistry(questionId);
        if (!question || !question.itensRegistry) {
            console.log('Questao nao encontrada no registry para toggle correct option');
            return;
        }
        
        // Encontrar item no itensRegistry
        const currentItem = question.itensRegistry.find(item => item.id === optionId);
        if (!currentItem) {
            console.log('Item nao encontrado no itensRegistry');
            return;
        }
        
        // Verificar se esta opção já está marcada como correta
        const isCurrentlyCorrect = currentItem.isCorrect;
        
        if (isCurrentlyCorrect) {
            // Desmarcar esta opção no registry
            currentItem.isCorrect = false;
            
            // Remover área de justificativa da questão se existir
            const justificationArea = questionBlock.querySelector('.justification-area');
            if (justificationArea) {
                justificationArea.remove();
            }
            
            console.log('Opcao desmarcada como correta');
        } else {
            // Primeiro, desmarcar todas as outras opções no registry
            question.itensRegistry.forEach(item => {
                item.isCorrect = false;
            });
            
            // Marcar esta opção como correta no registry
            currentItem.isCorrect = true;
            
            // Remover área de justificativa existente da questão
            const existingJustificationArea = questionBlock.querySelector('.justification-area');
            if (existingJustificationArea) {
                existingJustificationArea.remove();
            }
            
            // Adicionar área de justificativa
            this.addJustificationArea(optionItem);
            
            console.log('Opcao marcada como correta');
        }
        
        // Re-renderizar opções para atualizar visual
        this.renderQuestionOptions(question);
        
        // Atualizar dados da questão
        this.updateQuestionCorrectAnswer(questionBlock);
    }
    
    updateQuestionCorrectAnswer(questionBlock) {
        // Esta função será usada para sincronizar com o sistema de questões avançado
        const correctOption = questionBlock.querySelector('.option-item.correct-option');
        const questionId = questionBlock.dataset.questionId;
        
        if (correctOption) {
            const optionId = correctOption.dataset.optionId;
            console.log(`Questao ${questionId}: Resposta correta e opcao ${optionId}`);
        } else {
            console.log(`Questao ${questionId}: Nenhuma resposta correta selecionada`);
        }
    }

    addJustificationArea(optionItem) {
        const questionBlock = optionItem.closest('.question-block');
        const questionFooter = questionBlock.querySelector('.question-footer');
        
        const questionId = questionBlock.dataset.questionId;
        const question = this.questionsRegistry?.find(q => q.id === questionId);
        
        // Se já existe justificativa salva, mostrar visualização compacta
        if (question && question.justificativa && question.justificativa.trim()) {
            // Verificar se já existe visualização compacta
            const existingCompact = questionBlock.querySelector('.justification-compact');
            if (!existingCompact) {
                this.renderJustificationCompact(questionBlock, question.justificativa);
                this.disableCorrectOptionCheck(questionBlock);
            }
            return;
        }
        
        // Verificar se já existe uma área de justificativa na questão
        const existingJustification = questionBlock.querySelector('.justification-area');
        if (existingJustification) {
            // Se já existe, apenas focar na textarea
            const textarea = existingJustification.querySelector('.justification-input');
            setTimeout(() => {
                textarea.focus();
            }, 100);
            return;
        }
        
        // Criar área de justificativa
        const justificationHTML = `
            <div class="justification-area">
                <label class="justification-label">
                    <i class="material-icons">comment</i>
                    Justificativa para a resposta correta:
                </label>
                <div style="display: flex; align-items: flex-end; gap: 8px;">
                    <textarea 
                        class="justification-input" 
                        placeholder="Explique por que esta é a resposta correta..."
                        rows="3"
                        style="flex: 1; resize: vertical; padding: 8px; border: 1px solid var(--md-sys-color-outline); border-radius: 4px;"
                    ></textarea>
                    <button class="justification-save" title="Salvar justificativa" 
                            style="width: 32px; height: 32px; border-radius: 50%; border: 1px solid var(--md-sys-color-outline); background: var(--md-sys-color-surface); color: var(--md-sys-color-on-surface); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 16px;">✓</button>
                    <button class="justification-cancel" title="Cancelar"
                            style="width: 32px; height: 32px; border-radius: 50%; border: 1px solid var(--md-sys-color-outline); background: var(--md-sys-color-surface); color: var(--md-sys-color-on-surface); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 16px;">×</button>
                </div>
                <div class="justification-hint">
                    <i class="material-icons">info</i>
                    Esta justificativa será mostrada aos usuários após responderem
                </div>
            </div>
        `;
        
        // Inserir a área de justificativa após o question-footer
        questionFooter.insertAdjacentHTML('afterend', justificationHTML);
        
        // Focar na textarea e adicionar atalhos de teclado
        const textarea = questionBlock.querySelector('.justification-input');
        
        // Carregar justificativa existente se houver (usando a variável question já declarada)
        if (question && question.justificativa) {
            textarea.value = question.justificativa;
        }
        
        // Adicionar event listener para atalhos de teclado
        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                // Ctrl+Enter para salvar
                e.preventDefault();
                this.saveJustification(questionBlock);
            } else if (e.key === 'Escape') {
                // Escape para cancelar
                e.preventDefault();
                this.cancelJustification(questionBlock);
            }
        });
        
        setTimeout(() => {
            textarea.focus();
        }, 100);
        
        console.log('Area de justificativa adicionada a questao');
    }
    
    /**
     * Salva a justificativa da questão
     * @param {HTMLElement} questionBlock - Bloco da questão
     */
    saveJustification(questionBlock) {
        const questionId = questionBlock.dataset.questionId;
        const justificationArea = questionBlock.querySelector('.justification-area');
        const textarea = justificationArea?.querySelector('.justification-input');
        
        if (!textarea) return;
        
        const justificationText = textarea.value.trim();
        
        // Não salvar se texto estiver vazio
        if (!justificationText) {
            this.cancelJustification(questionBlock);
            return;
        }
        
        // Salvar no registry
        this.setQuestionJustificativa(questionId, justificationText);
        
        // Substituir área de edição pela visualização compacta
        this.renderJustificationCompact(questionBlock, justificationText);
        
        // Desabilitar botão check da opção correta
        this.disableCorrectOptionCheck(questionBlock);
        
        console.log(`Justificativa salva para questao ${questionId}: "${justificationText}"`);
    }
    
    /**
     * Cancela a edição da justificativa
     * @param {HTMLElement} questionBlock - Bloco da questão
     */
    cancelJustification(questionBlock) {
        const justificationArea = questionBlock.querySelector('.justification-area');
        
        if (justificationArea) {
            justificationArea.remove();
        }
        
        console.log('Edicao de justificativa cancelada');
    }
    
    /**
     * Renderiza a justificativa em formato compacto com controles
     * @param {HTMLElement} questionBlock - Bloco da questão
     * @param {string} justificationText - Texto da justificativa
     */
    renderJustificationCompact(questionBlock, justificationText) {
        const questionFooter = questionBlock.querySelector('.question-footer');
        if (!questionFooter) return;
        
        // Remover área de edição existente se houver
        const existingArea = questionBlock.querySelector('.justification-area');
        if (existingArea) {
            existingArea.remove();
        }
        
        // Criar visualização compacta
        const compactHTML = `
            <div class="justification-compact">
                <div style="display: flex; align-items: center; gap: 8px; padding: 8px; background: var(--md-sys-color-surface-container-lowest); border: 1px solid var(--md-sys-color-outline-variant); border-radius: 4px;">
                    <i class="material-icons" style="color: var(--md-sys-color-on-surface-variant); font-size: 18px;">comment</i>
                    <span class="justification-text" style="flex: 1; font-size: 14px; color: var(--md-sys-color-on-surface); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${justificationText}</span>
                    <button class="justification-edit" title="Editar justificativa" 
                            style="width: 24px; height: 24px; border-radius: 50%; border: 1px solid var(--md-sys-color-outline); background: var(--md-sys-color-surface); color: var(--md-sys-color-on-surface); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">✎</button>
                    <button class="justification-remove" title="Remover justificativa"
                            style="width: 24px; height: 24px; border-radius: 50%; border: 1px solid var(--md-sys-color-outline); background: var(--md-sys-color-surface); color: var(--md-sys-color-on-surface); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px;">×</button>
                </div>
            </div>
        `;
        
        // Inserir após o question-footer
        questionFooter.insertAdjacentHTML('afterend', compactHTML);
    }
    
    /**
     * Desabilita o botão check da opção correta
     * @param {HTMLElement} questionBlock - Bloco da questão
     */
    disableCorrectOptionCheck(questionBlock) {
        const correctOption = questionBlock.querySelector('.option-item.correct-option');
        if (correctOption) {
            const checkButton = correctOption.querySelector('.check-option');
            if (checkButton) {
                checkButton.disabled = true;
                checkButton.style.opacity = '0.5';
                checkButton.style.cursor = 'not-allowed';
            }
        }
    }
    
    /**
     * Edita uma justificativa existente
     * @param {HTMLElement} questionBlock - Bloco da questão
     */
    editJustification(questionBlock) {
        const questionId = questionBlock.dataset.questionId;
        const question = this.questionsRegistry?.find(q => q.id === questionId);
        
        if (!question || !question.justificativa) return;
        
        // Remover visualização compacta
        const compactArea = questionBlock.querySelector('.justification-compact');
        if (compactArea) {
            compactArea.remove();
        }
        
        // Reabilitar botão check
        this.enableCorrectOptionCheck(questionBlock);
        
        // Abrir área de edição com texto existente
        this.addJustificationArea(questionBlock);
    }
    
    /**
     * Remove uma justificativa existente
     * @param {HTMLElement} questionBlock - Bloco da questão
     */
    removeJustification(questionBlock) {
        const questionId = questionBlock.dataset.questionId;
        
        // Limpar justificativa do registry
        this.setQuestionJustificativa(questionId, '');
        
        // Remover visualização compacta
        const compactArea = questionBlock.querySelector('.justification-compact');
        if (compactArea) {
            compactArea.remove();
        }
        
        // Reabilitar botão check
        this.enableCorrectOptionCheck(questionBlock);
        
        console.log(`Justificativa removida da questao ${questionId}`);
    }
    
    /**
     * Reabilita o botão check da opção correta
     * @param {HTMLElement} questionBlock - Bloco da questão
     */
    enableCorrectOptionCheck(questionBlock) {
        const correctOption = questionBlock.querySelector('.option-item.correct-option');
        if (correctOption) {
            const checkButton = correctOption.querySelector('.check-option');
            if (checkButton) {
                checkButton.disabled = false;
                checkButton.style.opacity = '';
                checkButton.style.cursor = '';
            }
        }
    }
    
    // === SISTEMA DE FORMULARIOS MULTIPLOS ===
    
    /**
     * Cria um novo formRegistry
     * @param {string} name - Nome do formulário
     * @param {string} description - Descrição do formulário  
     * @param {string} icon - Ícone do Material Icons
     * @returns {Object} formRegistry criado
     */
    createFormRegistry(name = 'Novo Formulario', description = '', icon = 'quiz') {
        const formId = `form_${Date.now()}`;
        const now = new Date();
        
        const formRegistry = {
            id: formId,
            name: name,
            description: description,
            icon: icon,
            created: now,
            modified: now,
            status: 'draft',
            
            // Configurações do formulário
            settings: {
                timeLimit: 0,           // 0 = sem limite
                shuffleQuestions: false,
                showResults: true,
                allowRetake: false,
                collectEmail: false
            },
            
            // Registry de questões (inicialmente vazio)
            questionsRegistry: [],
            
            // Metadados
            metadata: {
                totalQuestions: 0,
                estimatedTime: 0,
                lastAccessed: now
            }
        };
        
        // Adicionar ao formsRegistry
        this.formsRegistry.set(formId, formRegistry);
        
        // Sincronizar dados entre páginas
        this.syncFormsData();
        
        console.log(`FormRegistry criado: ${formId} - ${name}`);
        return formRegistry;
    }
    
    /**
     * Ativa um formulário específico
     * @param {string} formId - ID do formulário a ser ativado
     */
    activateForm(formId) {
        const formRegistry = this.formsRegistry.get(formId);
        if (!formRegistry) {
            console.log(`Formulario nao encontrado: ${formId}`);
            return false;
        }
        
        // Atualizar contexto atual
        this.currentFormId = formId;
        this.questionsRegistry = formRegistry.questionsRegistry;
        
        // Atualizar último acesso
        formRegistry.metadata.lastAccessed = new Date();
        
        // Atualizar visual na sidebar
        this.updateFormCardStates();
        
        console.log(`Formulario ativado: ${formId} - ${formRegistry.name}`);
        return true;
    }
    
    /**
     * Obtém o formRegistry do formulário ativo
     * @returns {Object|null} formRegistry ativo ou null
     */
    getCurrentFormRegistry() {
        if (!this.currentFormId) return null;
        return this.formsRegistry.get(this.currentFormId);
    }
    
    /**
     * Atualiza metadados do formulário ativo
     */
    updateCurrentFormMetadata() {
        const formRegistry = this.getCurrentFormRegistry();
        if (formRegistry) {
            formRegistry.metadata.totalQuestions = formRegistry.questionsRegistry.length;
            formRegistry.metadata.estimatedTime = Math.max(1, Math.ceil(formRegistry.metadata.totalQuestions * 2)); // 2 min por questão
            formRegistry.modified = new Date();
        }
    }
    
    
    /**
     * Gera HTML para card de questão no clipboard
     */
    generateClipboardQuestionCardHTML(question, index) {
        const typeNames = {
            'multiple-choice': 'Múltipla escolha',
            'checkboxes': 'Caixas de seleção',
            'short-answer': 'Resposta curta',
            'paragraph': 'Resposta longa - Texto',
            'long-answer': 'Resposta longa - Matemática e código',
            'file-upload': 'Upload de arquivo',
            'section': 'Seção',
            'video': 'Vídeo',
            'image': 'Imagem'
        };
        
        const typeName = typeNames[question.type] || 'Questão';
        const enunciado = question.enunciado || 'Sem enunciado';
        const truncatedEnunciado = enunciado.length > 50 ? enunciado.substring(0, 50) + '...' : enunciado;
        
        return `
            <div class="clipboard-question-card" data-clipboard-index="${index}">
                <div class="clipboard-question-header">
                    <i class="material-icons">quiz</i>
                    <span class="clipboard-question-type">${typeName}</span>
                </div>
                <div class="clipboard-question-content">
                    <div class="clipboard-question-enunciado">${truncatedEnunciado}</div>
                </div>
            </div>
        `;
    }
    
    /**
     * Renderiza cards dos formulários na sidebar
     */
    renderFormCards() {
        const container = document.getElementById('forms-in-memory');
        if (!container) return;
        
        let html = '';
        
        // Renderizar questões do clipboard se houver
        if (this.questionsClipboard.length > 0) {
            html += '<div class="clipboard-section">';
            html += '<div class="clipboard-header">Questões copiadas:</div>';
            
            this.questionsClipboard.forEach((question, index) => {
                html += this.generateClipboardQuestionCardHTML(question, index);
            });
            
            html += '</div>';
            html += '<div class="forms-section-header">Formulários:</div>';
        }
        
        // Renderizar formulários
        if (this.formsRegistry.size === 0) {
            html += '<div style="color: var(--md-sys-color-on-surface-variant); font-size: 12px; text-align: center; padding: var(--spacing-md);">Nenhum formulario criado</div>';
        } else {
            const cardsHTML = Array.from(this.formsRegistry.values())
                .sort((a, b) => b.metadata.lastAccessed - a.metadata.lastAccessed) // Mais recente primeiro
                .map(form => this.generateFormCardHTML(form))
                .join('');
            html += cardsHTML;
        }
        
        container.innerHTML = html;
        
        // Adicionar event listeners
        this.setupFormCardListeners();
        this.setupClipboardCardListeners();
        
        // Renderizar ícones na header para modo responsivo
        this.renderHeaderFormIcons();
    }
    
    /**
     * Gera HTML para um card de formulário
     * @param {Object} formRegistry - Dados do formulário
     * @returns {string} HTML do card
     */
    generateFormCardHTML(formRegistry) {
        const isActive = formRegistry.id === this.currentFormId;
        const questionsCount = formRegistry.metadata.totalQuestions;
        const timeText = formRegistry.metadata.estimatedTime > 0 ? `~${formRegistry.metadata.estimatedTime}min` : '';
        
        return `
            <div class="form-card ${isActive ? 'active' : ''}" data-form-id="${formRegistry.id}">
                <i class="material-icons form-card-icon">${formRegistry.icon}</i>
                <div class="form-card-content">
                    <div class="form-card-name">${formRegistry.name}</div>
                    <div class="form-card-info">${questionsCount} questoes ${timeText}</div>
                </div>
                <div class="form-card-actions">
                    <button class="form-card-action" data-action="edit" title="Editar">
                        <i class="material-icons">edit</i>
                    </button>
                    <button class="form-card-action" data-action="delete" title="Excluir">
                        <i class="material-icons">delete</i>
                    </button>
                </div>
            </div>
        `;
    }
    
    /**
     * Configura event listeners para os cards de formulários
     */
    setupFormCardListeners() {
        const cards = document.querySelectorAll('.form-card');
        
        cards.forEach(card => {
            const formId = card.dataset.formId;
            
            // Click no card para carregar formulário existente
            card.addEventListener('click', (e) => {
                if (e.target.closest('.form-card-action')) return; // Ignorar clicks nos botões de ação
                
                this.loadExistingForm(formId);
            });
            
            // Botões de ação
            const editBtn = card.querySelector('[data-action="edit"]');
            const deleteBtn = card.querySelector('[data-action="delete"]');
            
            if (editBtn) {
                editBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.editFormName(formId);
                });
            }
            
            if (deleteBtn) {
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.deleteForm(formId);
                });
            }
        });
    }
    
    /**
     * Configura event listeners para os cards de questões do clipboard
     */
    setupClipboardCardListeners() {
        const clipboardCards = document.querySelectorAll('.clipboard-question-card');
        
        clipboardCards.forEach(card => {
            const clipboardIndex = parseInt(card.dataset.clipboardIndex);
            
            // Click no card para colar questão no formulário ativo
            card.addEventListener('click', (e) => {
                this.pasteQuestionFromClipboard(clipboardIndex);
            });
        });
    }
    
    /**
     * Copia uma questão para o clipboard
     * @param {HTMLElement} questionBlock - Elemento DOM da questão
     */
    copyQuestionToClipboard(questionBlock) {
        const questionId = questionBlock.dataset.questionId;
        
        // Buscar questão no registry
        const questionInRegistry = this.questionsRegistry.find(q => q.id === questionId);
        if (!questionInRegistry) {
            console.warn(`Questao ${questionId} nao encontrada no registry`);
            return;
        }
        
        // Sincronizar dados do DOM para o registry antes de copiar
        this.syncQuestionDataFromDOMToRegistry(questionId);
        
        // Criar cópia profunda da questão (sem referência ao DOM)
        const questionCopy = {
            type: questionInRegistry.type,
            enunciado: questionInRegistry.enunciado,
            justificativa: questionInRegistry.justificativa,
            rubrica: questionInRegistry.rubrica,
            itensRegistry: questionInRegistry.itensRegistry ? 
                JSON.parse(JSON.stringify(questionInRegistry.itensRegistry)) : null,
            description: questionInRegistry.description || '',
            videoUrl: questionInRegistry.videoUrl || '',
            imageUrl: questionInRegistry.imageUrl || '',
            copiedAt: new Date(),
            originalId: questionId
        };
        
        // Adicionar ao clipboard
        this.questionsClipboard.push(questionCopy);
        
        // Atualizar sidebar
        this.renderFormCards();
        
        console.log(`Questao ${questionId} copiada para clipboard`, questionCopy);
    }
    
    /**
     * Cola uma questão do clipboard no formulário ativo
     * @param {number} clipboardIndex - Índice da questão no clipboard
     */
    pasteQuestionFromClipboard(clipboardIndex) {
        if (!this.currentFormId || !this.questionsRegistry) {
            console.warn('Nenhum formulario ativo para colar a questao');
            return;
        }
        
        const questionToPaste = this.questionsClipboard[clipboardIndex];
        if (!questionToPaste) {
            console.warn(`Questao no indice ${clipboardIndex} nao encontrada no clipboard`);
            return;
        }
        
        // Determinar posição de inserção (abaixo da questão em foco)
        const focusedQuestion = this.questionsRegistry.find(q => q.focus === true);
        const insertPosition = focusedQuestion ? 
            this.questionsRegistry.findIndex(q => q.id === focusedQuestion.id) + 1 : 
            this.questionsRegistry.length;
        
        // Gerar novo ID e número para a questão
        const newQuestionId = `q_${Date.now()}`;
        const newQuestionNumber = this.questionsRegistry.length + 1;
        
        // Criar nova questão baseada na do clipboard
        const newQuestionData = {
            id: newQuestionId,
            number: newQuestionNumber,
            type: questionToPaste.type,
            title: this.generateQuestionTitle(newQuestionNumber, questionToPaste.type),
            itensRegistry: questionToPaste.itensRegistry || []
        };
        
        // Adicionar ao registry na posição correta
        this.questionsRegistry.splice(insertPosition, 0, {
            id: newQuestionId,
            type: questionToPaste.type,
            position: insertPosition,
            order: newQuestionNumber,
            focus: true,
            enunciado: questionToPaste.enunciado || '',
            justificativa: questionToPaste.justificativa || '',
            rubrica: questionToPaste.rubrica || '',
            selecionado: false,
            rubricaform: false,
            itensRegistry: questionToPaste.itensRegistry || [],
            description: questionToPaste.description || '',
            videoUrl: questionToPaste.videoUrl || '',
            imageUrl: questionToPaste.imageUrl || '',
            domElement: null
        });
        
        // Remover foco de outras questões
        this.questionsRegistry.forEach(q => {
            if (q.id !== newQuestionId) q.focus = false;
        });
        
        // Atualizar posições e ordens
        this.updateQuestionsOrder();
        
        // Criar HTML da questão
        this.createQuestionHTML(newQuestionData, insertPosition);
        
        // Renderizar opções se for questão com itens
        if (questionToPaste.type === 'multiple-choice' || questionToPaste.type === 'checkboxes') {
            setTimeout(() => {
                const newQuestion = this.questionsRegistry.find(q => q.id === newQuestionId);
                if (newQuestion) {
                    this.renderQuestionOptions(newQuestion);
                    
                    // Re-renderizar justificativa compacta se existir
                    if (newQuestion.justificativa && newQuestion.justificativa.trim()) {
                        const questionBlock = document.querySelector(`[data-question-id="${newQuestionId}"]`);
                        if (questionBlock) {
                            const existingCompact = questionBlock.querySelector('.justification-compact');
                            if (!existingCompact) {
                                this.renderJustificationCompact(questionBlock, newQuestion.justificativa);
                                this.disableCorrectOptionCheck(questionBlock);
                            }
                        }
                    }
                }
            }, 100);
        }
        
        // Restaurar dados de vídeo se for um bloco de vídeo
        if (questionToPaste.type === 'video') {
            setTimeout(() => {
                // Restaurar descrição
                const descriptionField = document.getElementById(`video-description-${newQuestionId}`);
                if (descriptionField && questionToPaste.description) {
                    descriptionField.value = questionToPaste.description;
                }
                
                // Restaurar URL
                const urlField = document.getElementById(`video-url-${newQuestionId}`);
                if (urlField && questionToPaste.videoUrl) {
                    urlField.value = questionToPaste.videoUrl;
                    
                    // Atualizar preview automaticamente
                    this.updateVideoPreview(newQuestionId, questionToPaste.videoUrl);
                }
            }, 100);
        }
        
        // Restaurar dados de imagem se for um bloco de imagem
        if (questionToPaste.type === 'image') {
            setTimeout(() => {
                // Restaurar descrição
                const descriptionField = document.getElementById(`image-description-${newQuestionId}`);
                if (descriptionField && questionToPaste.description) {
                    descriptionField.value = questionToPaste.description;
                }
                
                // Restaurar URL
                const urlField = document.getElementById(`image-url-${newQuestionId}`);
                if (urlField && questionToPaste.imageUrl) {
                    urlField.value = questionToPaste.imageUrl;
                    
                    // Atualizar preview automaticamente
                    this.updateImagePreview(newQuestionId, questionToPaste.imageUrl);
                }
            }, 100);
        }
        
        // Remover questão do clipboard
        this.questionsClipboard.splice(clipboardIndex, 1);
        
        // Atualizar metadados do formulário e sidebar
        this.updateCurrentFormMetadata();
        this.renderFormCards();
        
        console.log(`Questao colada do clipboard: ${newQuestionId} na posicao ${insertPosition}`);
    }
    
    /**
     * Atualiza estados visuais dos cards (ativo/inativo)
     */
    updateFormCardStates() {
        const cards = document.querySelectorAll('.form-card');
        cards.forEach(card => {
            const isActive = card.dataset.formId === this.currentFormId;
            card.classList.toggle('active', isActive);
        });
    }
    
    /**
     * Edita nome de um formulário
     * @param {string} formId - ID do formulário
     */
    editFormName(formId) {
        const formRegistry = this.formsRegistry.get(formId);
        if (!formRegistry) return;
        
        const newName = prompt('Novo nome do formulario:', formRegistry.name);
        if (newName && newName.trim() && newName !== formRegistry.name) {
            formRegistry.name = newName.trim();
            formRegistry.modified = new Date();
            this.renderFormCards();
            console.log(`Nome do formulario alterado: ${formId} -> ${newName}`);
        }
    }
    
    /**
     * Exclui um formulário
     * @param {string} formId - ID do formulário
     */
    deleteForm(formId) {
        const formRegistry = this.formsRegistry.get(formId);
        if (!formRegistry) return;
        
        if (confirm(`Excluir formulario "${formRegistry.name}"?\nEsta acao nao pode ser desfeita.`)) {
            this.formsRegistry.delete(formId);
            
            // Se o formulário excluído era o ativo, limpar contexto
            if (this.currentFormId === formId) {
                this.currentFormId = null;
                this.questionsRegistry = null;
            }
            
            this.syncFormsData();
            console.log(`Formulario excluido: ${formId} - ${formRegistry.name}`);
        }
    }
    
    /**
     * Sincroniza dados de formulários entre páginas
     * Atualiza contador na homepage e cards na página de formulários
     */
    syncFormsData() {
        // Sincronizar formsRegistry com window.KyrionForms.forms
        window.KyrionForms.forms = Array.from(this.formsRegistry.values());
        
        // Atualizar contador na homepage se estiver visível
        const statCards = document.querySelectorAll('.stat-card');
        statCards.forEach(card => {
            const statLabel = card.querySelector('.stat-label');
            const statNumber = card.querySelector('.stat-number');
            if (statLabel && statNumber && statLabel.textContent === 'Formulários') {
                statNumber.textContent = window.KyrionForms.forms.length;
            }
        });
        
        // Se estiver na página "Meus Formulários", recriar a página
        if (window.KyrionForms.currentRoute === 'forms') {
            this.loadFormsOverview();
        } else if (window.KyrionForms.currentRoute === 'forms-created') {
            this.loadForms();
        }
        
        // Renderizar cards na sidebar sempre
        this.renderFormCards();
    }
    
    /**
     * Renderiza ícones dos formulários na header para modo responsivo
     */
    renderHeaderFormIcons() {
        const container = document.getElementById('header-forms-icons');
        if (!container) return;
        
        if (this.formsRegistry.size === 0) {
            container.innerHTML = '';
            return;
        }
        
        const iconsHTML = Array.from(this.formsRegistry.values())
            .sort((a, b) => b.metadata.lastAccessed - a.metadata.lastAccessed) // Mais recente primeiro
            .slice(0, 5) // Máximo 5 ícones na header
            .map(form => this.generateHeaderFormIcon(form))
            .join('');
            
        container.innerHTML = iconsHTML;
        
        // Adicionar event listeners para os ícones
        this.setupHeaderFormIconListeners();
    }
    
    /**
     * Gera HTML para um ícone de formulário na header
     * @param {Object} formRegistry - Dados do formulário
     * @returns {string} HTML do ícone
     */
    generateHeaderFormIcon(formRegistry) {
        const isActive = formRegistry.id === this.currentFormId;
        
        return `
            <div class="header-form-icon ${isActive ? 'active' : ''}" data-form-id="${formRegistry.id}">
                <i class="material-icons">${formRegistry.icon}</i>
                <div class="header-form-tooltip">
                    <span class="tooltip-title">${formRegistry.name}</span>
                    <span class="tooltip-description">${formRegistry.description}</span>
                </div>
            </div>
        `;
    }
    
    /**
     * Configura event listeners para os ícones da header
     */
    setupHeaderFormIconListeners() {
        const icons = document.querySelectorAll('.header-form-icon');
        
        icons.forEach(icon => {
            const formId = icon.dataset.formId;
            
            icon.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                // Carregar formulário existente
                this.loadExistingForm(formId);
            });
        });
    }
    
    updateTheme() {
        const theme = window.KyrionForms.settings.theme;
        const isDark = theme === 'dark' || 
                      (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches);
        
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        console.log(`Tema aplicado: ${isDark ? 'dark' : 'light'}`);
    }
}

// Inicializar a aplicação quando o DOM estiver carregado
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM carregado, inicializando aplicacao...');
    window.simpleApp = new SimpleKyrionApp();
    
    // Expor funções de edição de enunciado globalmente
    window.editQuestionStatement = (questionId) => window.simpleApp.editQuestionStatement(questionId);
    window.saveQuestionStatement = (questionId) => window.simpleApp.saveQuestionStatement(questionId);
    window.handleStatementKeydown = (event, questionId) => window.simpleApp.handleStatementKeydown(event, questionId);
    
    // Expor função de debug globalmente para facilitar diagnóstico
    window.debugApp = () => {
        if (window.simpleApp) {
            window.simpleApp.debugRegistries();
        } else {
            console.log('App nao inicializado');
        }
    };
    
    console.log('Funcoes de edicao de enunciado expostas globalmente');
});
