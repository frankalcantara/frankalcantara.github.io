/**
 * Sistema de Roteamento SPA para Kyrion Forms
 * Gerencia navegação entre diferentes views da aplicação
 */

class Router {
    constructor() {
        this.routes = new Map();
        this.currentRoute = null;
        this.contentElement = document.getElementById('content');
        this.loadingTemplate = this.createLoadingTemplate();
        
        this.setupHashNavigation();
    }
    
    setupHashNavigation() {
        // Escutar mudanças no hash
        window.addEventListener('hashchange', () => {
            this.handleRouteChange();
        });
        
        // Escutar navegação do browser
        window.addEventListener('popstate', () => {
            this.handleRouteChange();
        });
    }
    
    addRoute(path, config) {
        this.routes.set(path, {
            path,
            component: config.component,
            title: config.title || 'Kyrion Forms',
            beforeEnter: config.beforeEnter,
            afterEnter: config.afterEnter
        });
    }
    
    async navigate(path, data = {}) {
        try {
            // Atualizar hash se necessário
            const currentHash = window.location.hash.slice(1);
            if (currentHash !== path) {
                window.location.hash = path;
            }
            
            await this.loadRoute(path, data);
        } catch (error) {
            console.error('Erro na navegação:', error);
            this.showError('Erro ao carregar página');
        }
    }
    
    async loadRoute(path, data = {}) {
        // Extrair parâmetros da rota
        const { route, params } = this.matchRoute(path);
        
        if (!route) {
            console.warn(`Rota não encontrada: ${path}`);
            await this.loadRoute('home');
            return;
        }
        
        try {
            // Mostrar loading
            this.showLoading();
            
            // Executar beforeEnter se existir
            if (route.beforeEnter) {
                const canEnter = await route.beforeEnter(params, data);
                if (!canEnter) {
                    return;
                }
            }
            
            // Carregar componente
            const componentModule = await route.component();
            const ComponentClass = componentModule.default || componentModule;
            
            // Criar instância do componente
            const component = new ComponentClass({
                params,
                data,
                router: this
            });
            
            // Renderizar componente
            await this.renderComponent(component);
            
            // Atualizar estado
            this.currentRoute = route;
            window.KyrionForms.currentRoute = path;
            
            // Atualizar título da página
            document.title = route.title;
            
            // Atualizar navegação ativa
            this.updateActiveNavigation(path);
            
            // Executar afterEnter se existir
            if (route.afterEnter) {
                await route.afterEnter(params, data);
            }
            
        } catch (error) {
            console.error('Erro ao carregar rota:', error);
            this.showError('Erro ao carregar página');
            
            // Fallback para home
            if (path !== 'home') {
                await this.loadRoute('home');
            }
        }
    }
    
    matchRoute(path) {
        // Primeiro, tentar match exato
        if (this.routes.has(path)) {
            return {
                route: this.routes.get(path),
                params: {}
            };
        }
        
        // Tentar match com parâmetros
        for (const [routePath, route] of this.routes) {
            if (routePath.includes(':')) {
                const params = this.extractParams(routePath, path);
                if (params) {
                    return { route, params };
                }
            }
        }
        
        return { route: null, params: {} };
    }
    
    extractParams(routePath, actualPath) {
        const routeParts = routePath.split('/');
        const pathParts = actualPath.split('/');
        
        if (routeParts.length !== pathParts.length) {
            return null;
        }
        
        const params = {};
        
        for (let i = 0; i < routeParts.length; i++) {
            const routePart = routeParts[i];
            const pathPart = pathParts[i];
            
            if (routePart.startsWith(':')) {
                // Parâmetro dinâmico
                const paramName = routePart.slice(1);
                params[paramName] = pathPart;
            } else if (routePart !== pathPart) {
                // Parte estática não coincide
                return null;
            }
        }
        
        return params;
    }
    
    async renderComponent(component) {
        try {
            // Limpar conteúdo anterior
            this.contentElement.innerHTML = '';
            
            // Renderizar novo componente
            const html = await component.render();
            this.contentElement.innerHTML = html;
            
            // Executar scripts de inicialização do componente
            if (component.mount) {
                await component.mount();
            }
            
            // Animar entrada
            this.contentElement.classList.add('fade-in');
            
        } catch (error) {
            console.error('Erro ao renderizar componente:', error);
            throw error;
        }
    }
    
    showLoading() {
        this.contentElement.innerHTML = this.loadingTemplate;
    }
    
    showError(message) {
        this.contentElement.innerHTML = `
            <div class="error-container">
                <div class="error-icon">
                    <md-icon>error</md-icon>
                </div>
                <h2>Oops! Algo deu errado</h2>
                <p>${message}</p>
                <md-filled-button onclick="location.reload()">
                    Recarregar página
                </md-filled-button>
            </div>
        `;
    }
    
    createLoadingTemplate() {
        return `
            <div class="loading-container">
                <md-circular-progress indeterminate></md-circular-progress>
                <p>Carregando...</p>
            </div>
            <style>
                .loading-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 400px;
                    gap: var(--spacing-md);
                    color: var(--md-sys-color-on-surface-variant);
                }
                
                .error-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 400px;
                    gap: var(--spacing-md);
                    text-align: center;
                    padding: var(--spacing-xl);
                }
                
                .error-icon {
                    font-size: 64px;
                    color: var(--md-sys-color-error);
                }
                
                .error-container h2 {
                    color: var(--md-sys-color-on-surface);
                    margin: 0;
                }
                
                .error-container p {
                    color: var(--md-sys-color-on-surface-variant);
                    margin: 0;
                    max-width: 400px;
                }
            </style>
        `;
    }
    
    updateActiveNavigation(currentPath) {
        const navItems = document.querySelectorAll('[data-route]');
        
        navItems.forEach(item => {
            const route = item.getAttribute('data-route');
            
            // Remover classe ativa de todos
            item.classList.remove('active');
            
            // Adicionar classe ativa ao item correspondente
            if (this.isRouteActive(route, currentPath)) {
                item.classList.add('active');
            }
        });
    }
    
    isRouteActive(navRoute, currentPath) {
        // Match exato
        if (navRoute === currentPath) {
            return true;
        }
        
        // Match parcial para rotas aninhadas
        if (currentPath.startsWith(navRoute + '/')) {
            return true;
        }
        
        // Casos especiais
        if (navRoute === 'forms' && currentPath.startsWith('edit-form/')) {
            return true;
        }
        
        return false;
    }
    
    handleRouteChange() {
        const hash = window.location.hash.slice(1);
        const path = hash || 'home';
        
        if (path !== window.KyrionForms.currentRoute) {
            this.loadRoute(path);
        }
    }
    
    // Métodos utilitários para componentes
    goBack() {
        window.history.back();
    }
    
    replace(path, data = {}) {
        window.history.replaceState(data, '', `#${path}`);
        this.loadRoute(path, data);
    }
    
    getQueryParams() {
        const urlParams = new URLSearchParams(window.location.search);
        const params = {};
        
        for (const [key, value] of urlParams) {
            params[key] = value;
        }
        
        return params;
    }
    
    buildUrl(path, params = {}) {
        let url = `#${path}`;
        
        const queryString = new URLSearchParams(params).toString();
        if (queryString) {
            url += `?${queryString}`;
        }
        
        return url;
    }
}

// Classe base para componentes
class BaseComponent {
    constructor(options = {}) {
        this.params = options.params || {};
        this.data = options.data || {};
        this.router = options.router;
        this.element = null;
    }
    
    async render() {
        // Deve ser implementado pelas subclasses
        throw new Error('Método render() deve ser implementado');
    }
    
    async mount() {
        // Método opcional para inicialização após renderização
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Método opcional para configurar event listeners
    }
    
    unmount() {
        // Método opcional para limpeza antes da remoção
    }
    
    // Utilitários para manipulação do DOM
    $(selector) {
        return document.querySelector(selector);
    }
    
    $$(selector) {
        return document.querySelectorAll(selector);
    }
    
    createElement(tag, attributes = {}, content = '') {
        const element = document.createElement(tag);
        
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'innerHTML') {
                element.innerHTML = value;
            } else {
                element.setAttribute(key, value);
            }
        });
        
        if (content) {
            element.textContent = content;
        }
        
        return element;
    }
    
    // Utilitários para eventos
    emit(eventName, detail = {}) {
        const event = new CustomEvent(eventName, { detail });
        document.dispatchEvent(event);
    }
    
    on(eventName, handler) {
        document.addEventListener(eventName, handler);
    }
    
    off(eventName, handler) {
        document.removeEventListener(eventName, handler);
    }
    
    // Utilitários para dados
    async saveForm(form) {
        return window.app.saveForm(form);
    }
    
    getForm(id) {
        return window.app.getForm(id);
    }
    
    deleteForm(id) {
        return window.app.deleteForm(id);
    }
    
    async saveResponse(response) {
        return window.app.saveResponse(response);
    }
    
    getFormResponses(formId) {
        return window.app.getFormResponses(formId);
    }
    
    // Utilitários de UI
    showSuccess(message) {
        window.app.showSuccess(message);
    }
    
    showError(message) {
        window.app.showError(message);
    }
    
    formatDate(date) {
        return window.app.formatDate(date);
    }
    
    generateId() {
        return window.app.generateId();
    }
}

// Exportar classes
export { Router, BaseComponent };
