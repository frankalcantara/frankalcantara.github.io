/**
 * Kyrion Forms - Aplicação Principal
 * Sistema de formulários com suporte a LaTeX e código
 */

// Estado global da aplicação
window.KyrionForms = {
    currentRoute: 'home',
    currentForm: null,
    forms: [],
    responses: [],
    settings: {
        theme: 'light', // Forçar tema claro
        autoSave: true,
        autoSaveInterval: 30000 // 30 segundos
    }
};

// Classe principal da aplicação
class KyrionFormsApp {
    constructor() {
        this.router = null;
        this.storage = null;
        this.editors = new Map();
        this.autoSaveTimer = null;
        
        this.init();
    }
    
    async init() {
        try {
            // Inicializar storage
            this.storage = new StorageManager();
            await this.storage.init();
            
            // Carregar dados salvos
            await this.loadData();
            
            // Inicializar router
            this.router = new Router();
            this.setupRoutes();
            
            // Configurar event listeners
            this.setupEventListeners();
            
            // Inicializar interface
            this.setupUI();
            
            // Navegar para rota inicial
            this.router.navigate(this.getCurrentRoute());
            
            console.log('Kyrion Forms iniciado com sucesso');
        } catch (error) {
            console.error('Erro ao inicializar aplicação:', error);
            this.showError('Erro ao carregar a aplicação. Tente recarregar a página.');
        }
    }
    
    setupRoutes() {
        // Definir rotas da aplicação
        this.router.addRoute('home', {
            component: () => import('./views/home.js'),
            title: 'Início - Kyrion Forms'
        });
        
        this.router.addRoute('forms', {
            component: () => import('./views/forms.js'),
            title: 'Meus Formulários - Kyrion Forms'
        });
        
        this.router.addRoute('new-form', {
            component: () => import('./views/form-builder.js'),
            title: 'Novo Formulário - Kyrion Forms'
        });
        
        this.router.addRoute('edit-form/:id', {
            component: () => import('./views/form-builder.js'),
            title: 'Editar Formulário - Kyrion Forms'
        });
        
        this.router.addRoute('view-form/:id', {
            component: () => import('./views/form-viewer.js'),
            title: 'Visualizar Formulário - Kyrion Forms'
        });
        
        this.router.addRoute('respond-form/:id', {
            component: () => import('./views/form-response.js'),
            title: 'Responder Formulário - Kyrion Forms'
        });
        
        this.router.addRoute('responses', {
            component: () => import('./views/responses.js'),
            title: 'Respostas - Kyrion Forms'
        });
        
        this.router.addRoute('responses/:formId', {
            component: () => import('./views/form-responses.js'),
            title: 'Respostas do Formulário - Kyrion Forms'
        });
    }
    
    setupEventListeners() {
        // Menu mobile toggle
        const menuButton = document.getElementById('menu-button');
        const sidebar = document.getElementById('sidebar');
        
        if (menuButton && sidebar) {
            menuButton.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
            
            // Fechar sidebar ao clicar fora (mobile)
            document.addEventListener('click', (e) => {
                if (window.innerWidth <= 768 && 
                    !sidebar.contains(e.target) && 
                    !menuButton.contains(e.target)) {
                    sidebar.classList.remove('open');
                }
            });
        }
        
        // Navegação da sidebar
        const sidebarItems = document.querySelectorAll('[data-route]');
        sidebarItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const route = item.getAttribute('data-route');
                this.router.navigate(route);
                
                // Atualizar botão ativo
                sidebarItems.forEach(btn => btn.classList.remove('active'));
                item.classList.add('active');
                
                // Fechar sidebar em mobile
                if (window.innerWidth <= 768) {
                    sidebar.classList.remove('open');
                }
            });
        });
        
        // FAB principal
        const mainFab = document.getElementById('main-fab');
        if (mainFab) {
            mainFab.addEventListener('click', () => {
                this.router.navigate('new-form');
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + N - Novo formulário
            if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                e.preventDefault();
                this.router.navigate('new-form');
            }
            
            // Ctrl/Cmd + S - Salvar (se estivermos editando)
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.saveCurrentForm();
            }
        });
        
        // Auto-save
        if (window.KyrionForms.settings.autoSave) {
            this.startAutoSave();
        }
        
        // Detectar mudanças de tema do sistema
        if (window.matchMedia) {
            const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
            mediaQuery.addListener(() => {
                if (window.KyrionForms.settings.theme === 'auto') {
                    this.updateTheme();
                }
            });
        }
        
        // Salvar dados antes de sair
        window.addEventListener('beforeunload', () => {
            this.saveData();
        });
    }
    
    setupUI() {
        // Configurar tema inicial
        this.updateTheme();
        
        // Marcar item ativo na navegação
        this.updateActiveNavItem();
        
        // Configurar títulos dinâmicos
        this.updatePageTitle();
    }
    
    async loadData() {
        try {
            const data = await this.storage.getAllForms();
            window.KyrionForms.forms = data || [];
            
            const responses = await this.storage.getAllResponses();
            window.KyrionForms.responses = responses || [];
            
            const settings = await this.storage.getSettings();
            if (settings) {
                window.KyrionForms.settings = { ...window.KyrionForms.settings, ...settings };
            }
        } catch (error) {
            console.warn('Erro ao carregar dados:', error);
        }
    }
    
    async saveData() {
        try {
            await this.storage.saveForms(window.KyrionForms.forms);
            await this.storage.saveResponses(window.KyrionForms.responses);
            await this.storage.saveSettings(window.KyrionForms.settings);
        } catch (error) {
            console.error('Erro ao salvar dados:', error);
        }
    }
    
    startAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
        }
        
        this.autoSaveTimer = setInterval(() => {
            this.saveData();
        }, window.KyrionForms.settings.autoSaveInterval);
    }
    
    stopAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
            this.autoSaveTimer = null;
        }
    }
    
    saveCurrentForm() {
        // Disparar evento para salvar o formulário atual
        const event = new CustomEvent('save-current-form');
        document.dispatchEvent(event);
    }
    
    getCurrentRoute() {
        const hash = window.location.hash.slice(1);
        return hash || 'home';
    }
    
    updateActiveNavItem() {
        const currentRoute = window.KyrionForms.currentRoute;
        const navItems = document.querySelectorAll('[data-route]');
        
        navItems.forEach(item => {
            const route = item.getAttribute('data-route');
            if (route === currentRoute) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }
    
    updatePageTitle() {
        const route = this.router.currentRoute;
        if (route && route.title) {
            document.title = route.title;
        }
    }
    
    updateTheme() {
        const theme = window.KyrionForms.settings.theme;
        const isDark = theme === 'dark' || 
                      (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches);
        
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    }
    
    // Utilities
    generateId() {
        return 'form_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    formatDate(date) {
        if (!date) return '';
        if (typeof date === 'string') date = new Date(date);
        return date.toLocaleDateString('pt-BR', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    showError(message, duration = 5000) {
        this.showToast(message, 'error', duration);
    }
    
    showSuccess(message, duration = 3000) {
        this.showToast(message, 'success', duration);
    }
    
    showToast(message, type = 'info', duration = 3000) {
        // Remover toasts existentes
        const existingToasts = document.querySelectorAll('.toast');
        existingToasts.forEach(toast => toast.remove());
        
        // Criar novo toast
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        // Estilos
        Object.assign(toast.style, {
            position: 'fixed',
            top: '24px',
            right: '24px',
            padding: '16px 24px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '500',
            zIndex: '9999',
            maxWidth: '400px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease'
        });
        
        // Cores por tipo
        const colors = {
            error: '#d32f2f',
            success: '#2e7d32',
            warning: '#f57c00',
            info: '#1976d2'
        };
        
        toast.style.backgroundColor = colors[type] || colors.info;
        
        // Adicionar ao DOM
        document.body.appendChild(toast);
        
        // Animar entrada
        setTimeout(() => {
            toast.style.transform = 'translateX(0)';
        }, 100);
        
        // Remover após duração
        setTimeout(() => {
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
    }
    
    // API para outros módulos
    getForm(id) {
        return window.KyrionForms.forms.find(form => form.id === id);
    }
    
    saveForm(form) {
        const existingIndex = window.KyrionForms.forms.findIndex(f => f.id === form.id);
        
        if (existingIndex >= 0) {
            window.KyrionForms.forms[existingIndex] = form;
        } else {
            window.KyrionForms.forms.push(form);
        }
        
        this.saveData();
        return form;
    }
    
    deleteForm(id) {
        const index = window.KyrionForms.forms.findIndex(f => f.id === id);
        if (index >= 0) {
            window.KyrionForms.forms.splice(index, 1);
            this.saveData();
            return true;
        }
        return false;
    }
    
    getFormResponses(formId) {
        return window.KyrionForms.responses.filter(response => response.formId === formId);
    }
    
    saveResponse(response) {
        const existingIndex = window.KyrionForms.responses.findIndex(r => r.id === response.id);
        
        if (existingIndex >= 0) {
            window.KyrionForms.responses[existingIndex] = response;
        } else {
            window.KyrionForms.responses.push(response);
        }
        
        this.saveData();
        return response;
    }
}

// Storage Manager para gerenciar localStorage e IndexedDB
class StorageManager {
    constructor() {
        this.dbName = 'KyrionFormsDB';
        this.dbVersion = 1;
        this.db = null;
    }
    
    async init() {
        // Tentar usar IndexedDB, fallback para localStorage
        try {
            await this.initIndexedDB();
        } catch (error) {
            console.warn('IndexedDB não disponível, usando localStorage:', error);
            this.useLocalStorage = true;
        }
    }
    
    initIndexedDB() {
        return new Promise((resolve, reject) => {
            if (!window.indexedDB) {
                reject(new Error('IndexedDB não suportado'));
                return;
            }
            
            const request = indexedDB.open(this.dbName, this.dbVersion);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Store para formulários
                if (!db.objectStoreNames.contains('forms')) {
                    db.createObjectStore('forms', { keyPath: 'id' });
                }
                
                // Store para respostas
                if (!db.objectStoreNames.contains('responses')) {
                    const responseStore = db.createObjectStore('responses', { keyPath: 'id' });
                    responseStore.createIndex('formId', 'formId', { unique: false });
                }
                
                // Store para configurações
                if (!db.objectStoreNames.contains('settings')) {
                    db.createObjectStore('settings', { keyPath: 'key' });
                }
            };
        });
    }
    
    async getAllForms() {
        if (this.useLocalStorage) {
            const data = localStorage.getItem('kyrion_forms');
            return data ? JSON.parse(data) : [];
        }
        
        return this.getFromStore('forms');
    }
    
    async saveForms(forms) {
        if (this.useLocalStorage) {
            localStorage.setItem('kyrion_forms', JSON.stringify(forms));
            return;
        }
        
        const transaction = this.db.transaction(['forms'], 'readwrite');
        const store = transaction.objectStore('forms');
        
        // Limpar store
        await store.clear();
        
        // Adicionar todos os formulários
        for (const form of forms) {
            await store.add(form);
        }
    }
    
    async getAllResponses() {
        if (this.useLocalStorage) {
            const data = localStorage.getItem('kyrion_responses');
            return data ? JSON.parse(data) : [];
        }
        
        return this.getFromStore('responses');
    }
    
    async saveResponses(responses) {
        if (this.useLocalStorage) {
            localStorage.setItem('kyrion_responses', JSON.stringify(responses));
            return;
        }
        
        const transaction = this.db.transaction(['responses'], 'readwrite');
        const store = transaction.objectStore('responses');
        
        // Limpar store
        await store.clear();
        
        // Adicionar todas as respostas
        for (const response of responses) {
            await store.add(response);
        }
    }
    
    async getSettings() {
        if (this.useLocalStorage) {
            const data = localStorage.getItem('kyrion_settings');
            return data ? JSON.parse(data) : null;
        }
        
        const transaction = this.db.transaction(['settings'], 'readonly');
        const store = transaction.objectStore('settings');
        const request = store.get('app');
        
        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result?.value || null);
            request.onerror = () => reject(request.error);
        });
    }
    
    async saveSettings(settings) {
        if (this.useLocalStorage) {
            localStorage.setItem('kyrion_settings', JSON.stringify(settings));
            return;
        }
        
        const transaction = this.db.transaction(['settings'], 'readwrite');
        const store = transaction.objectStore('settings');
        await store.put({ key: 'app', value: settings });
    }
    
    async getFromStore(storeName) {
        const transaction = this.db.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.getAll();
        
        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }
}

// Inicializar aplicação quando DOM estiver carregado
document.addEventListener('DOMContentLoaded', () => {
    window.app = new KyrionFormsApp();
});

// Exportar para outros módulos
export { KyrionFormsApp, StorageManager };
