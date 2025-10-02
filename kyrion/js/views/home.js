/**
 * Home View - Página inicial do Kyrion Forms
 * Exibe dashboard com estatísticas e ações rápidas
 */

import { BaseComponent } from '../router.js';

export default class HomeView extends BaseComponent {
    constructor(options) {
        super(options);
        this.stats = {
            totalForms: 0,
            totalResponses: 0,
            recentForms: [],
            recentResponses: []
        };
    }
    
    async render() {
        // Calcular estatísticas
        this.calculateStats();
        
        return `
            <div class="home-container">
                <!-- Header da página -->
                <div class="page-header">
                    <h1>Bem-vindo ao Kyrion Forms</h1>
                    <p>Crie formulários inteligentes com suporte a LaTeX e código</p>
                </div>
                
                <!-- Cards de estatísticas -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon">
                            <md-icon>description</md-icon>
                        </div>
                        <div class="stat-content">
                            <div class="stat-number">${this.stats.totalForms}</div>
                            <div class="stat-label">Formulários</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <md-icon>analytics</md-icon>
                        </div>
                        <div class="stat-content">
                            <div class="stat-number">${this.stats.totalResponses}</div>
                            <div class="stat-label">Respostas</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <md-icon>trending_up</md-icon>
                        </div>
                        <div class="stat-content">
                            <div class="stat-number">${this.getAverageResponses()}</div>
                            <div class="stat-label">Média por Form</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">
                            <md-icon>schedule</md-icon>
                        </div>
                        <div class="stat-content">
                            <div class="stat-number">${this.getRecentFormsCount()}</div>
                            <div class="stat-label">Esta Semana</div>
                        </div>
                    </div>
                </div>
                
                <!-- Ações rápidas -->
                <div class="quick-actions">
                    <h2>Ações Rápidas</h2>
                    <div class="actions-grid">
                        <div class="action-card" data-action="new-form">
                            <div class="action-icon">
                                <md-icon>add_circle</md-icon>
                            </div>
                            <h3>Novo Formulário</h3>
                            <p>Crie um formulário com questões múltipla escolha e texto livre</p>
                        </div>
                        
                        <div class="action-card" data-action="view-forms">
                            <div class="action-icon">
                                <md-icon>folder</md-icon>
                            </div>
                            <h3>Meus Formulários</h3>
                            <p>Visualize e gerencie todos os seus formulários</p>
                        </div>
                        
                        <div class="action-card" data-action="view-responses">
                            <div class="action-icon">
                                <md-icon>assignment</md-icon>
                            </div>
                            <h3>Ver Respostas</h3>
                            <p>Analise as respostas recebidas dos formulários</p>
                        </div>
                        
                        <div class="action-card" data-action="import-form">
                            <div class="action-icon">
                                <md-icon>upload</md-icon>
                            </div>
                            <h3>Importar</h3>
                            <p>Importe formulários de outros sistemas</p>
                        </div>
                    </div>
                </div>
                
                <!-- Formulários recentes -->
                ${this.renderRecentForms()}
                
                <!-- Respostas recentes -->
                ${this.renderRecentResponses()}
                
                <!-- Features highlights -->
                <div class="features-section">
                    <h2>Recursos Avançados</h2>
                    <div class="features-grid">
                        <div class="feature-card">
                            <div class="feature-icon">
                                <md-icon>functions</md-icon>
                            </div>
                            <h3>Suporte LaTeX</h3>
                            <p>Crie questões com fórmulas matemáticas usando LaTeX</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">
                                <md-icon>code</md-icon>
                            </div>
                            <h3>Editor de Código</h3>
                            <p>Inclua snippets de código com syntax highlighting</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">
                                <md-icon>auto_awesome</md-icon>
                            </div>
                            <h3>Interface Intuitiva</h3>
                            <p>Design moderno e responsivo com Material Design</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">
                                <md-icon>cloud_sync</md-icon>
                            </div>
                            <h3>Auto-save</h3>
                            <p>Seus dados são salvos automaticamente enquanto você trabalha</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <style>
                .home-container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 0;
                }
                
                .page-header {
                    text-align: center;
                    margin-bottom: var(--spacing-xxl);
                    padding: var(--spacing-xl) 0;
                }
                
                .page-header h1 {
                    font-size: 2.5rem;
                    font-weight: 400;
                    color: var(--md-sys-color-on-background);
                    margin: 0 0 var(--spacing-md) 0;
                }
                
                .page-header p {
                    font-size: 1.2rem;
                    color: var(--md-sys-color-on-surface-variant);
                    margin: 0;
                }
                
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: var(--spacing-lg);
                    margin-bottom: var(--spacing-xxl);
                }
                
                .stat-card {
                    background: var(--md-sys-color-surface);
                    border: 1px solid var(--md-sys-color-outline-variant);
                    border-radius: var(--radius-md);
                    padding: var(--spacing-lg);
                    display: flex;
                    align-items: center;
                    gap: var(--spacing-md);
                    box-shadow: var(--elevation-1);
                    transition: box-shadow var(--transition-fast);
                }
                
                .stat-card:hover {
                    box-shadow: var(--elevation-2);
                }
                
                .stat-icon {
                    background: var(--md-sys-color-primary-container);
                    color: var(--md-sys-color-on-primary-container);
                    border-radius: var(--radius-lg);
                    width: 64px;
                    height: 64px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 32px;
                }
                
                .stat-content {
                    flex: 1;
                }
                
                .stat-number {
                    font-size: 2rem;
                    font-weight: 500;
                    color: var(--md-sys-color-on-surface);
                    line-height: 1;
                }
                
                .stat-label {
                    font-size: 0.9rem;
                    color: var(--md-sys-color-on-surface-variant);
                    margin-top: var(--spacing-xs);
                }
                
                .quick-actions {
                    margin-bottom: var(--spacing-xxl);
                }
                
                .quick-actions h2 {
                    font-size: 1.5rem;
                    font-weight: 500;
                    color: var(--md-sys-color-on-background);
                    margin: 0 0 var(--spacing-lg) 0;
                }
                
                .actions-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: var(--spacing-lg);
                }
                
                .action-card {
                    background: var(--md-sys-color-surface);
                    border: 1px solid var(--md-sys-color-outline-variant);
                    border-radius: var(--radius-md);
                    padding: var(--spacing-xl);
                    text-align: center;
                    cursor: pointer;
                    transition: all var(--transition-fast);
                    box-shadow: var(--elevation-1);
                }
                
                .action-card:hover {
                    box-shadow: var(--elevation-3);
                    transform: translateY(-2px);
                    border-color: var(--md-sys-color-primary);
                }
                
                .action-icon {
                    background: var(--md-sys-color-secondary-container);
                    color: var(--md-sys-color-on-secondary-container);
                    border-radius: var(--radius-xl);
                    width: 80px;
                    height: 80px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 40px;
                    margin: 0 auto var(--spacing-md);
                }
                
                .action-card h3 {
                    font-size: 1.2rem;
                    font-weight: 500;
                    color: var(--md-sys-color-on-surface);
                    margin: 0 0 var(--spacing-sm) 0;
                }
                
                .action-card p {
                    color: var(--md-sys-color-on-surface-variant);
                    margin: 0;
                    line-height: 1.5;
                }
                
                .recent-section {
                    margin-bottom: var(--spacing-xxl);
                }
                
                .recent-section h2 {
                    font-size: 1.5rem;
                    font-weight: 500;
                    color: var(--md-sys-color-on-background);
                    margin: 0 0 var(--spacing-lg) 0;
                }
                
                .recent-list {
                    display: flex;
                    flex-direction: column;
                    gap: var(--spacing-sm);
                }
                
                .recent-item {
                    background: var(--md-sys-color-surface);
                    border: 1px solid var(--md-sys-color-outline-variant);
                    border-radius: var(--radius-sm);
                    padding: var(--spacing-md);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    cursor: pointer;
                    transition: background-color var(--transition-fast);
                }
                
                .recent-item:hover {
                    background-color: var(--md-sys-color-surface-variant);
                }
                
                .recent-item-content {
                    flex: 1;
                }
                
                .recent-item-title {
                    font-weight: 500;
                    color: var(--md-sys-color-on-surface);
                    margin: 0 0 var(--spacing-xs) 0;
                }
                
                .recent-item-meta {
                    font-size: 0.9rem;
                    color: var(--md-sys-color-on-surface-variant);
                    margin: 0;
                }
                
                .features-section {
                    margin-bottom: var(--spacing-xxl);
                }
                
                .features-section h2 {
                    font-size: 1.5rem;
                    font-weight: 500;
                    color: var(--md-sys-color-on-background);
                    margin: 0 0 var(--spacing-lg) 0;
                    text-align: center;
                }
                
                .features-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: var(--spacing-lg);
                }
                
                .feature-card {
                    background: var(--md-sys-color-surface);
                    border: 1px solid var(--md-sys-color-outline-variant);
                    border-radius: var(--radius-md);
                    padding: var(--spacing-lg);
                    text-align: center;
                    box-shadow: var(--elevation-1);
                }
                
                .feature-icon {
                    background: var(--md-sys-color-tertiary-container);
                    color: var(--md-sys-color-on-tertiary-container);
                    border-radius: var(--radius-xl);
                    width: 64px;
                    height: 64px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 32px;
                    margin: 0 auto var(--spacing-md);
                }
                
                .feature-card h3 {
                    font-size: 1.1rem;
                    font-weight: 500;
                    color: var(--md-sys-color-on-surface);
                    margin: 0 0 var(--spacing-sm) 0;
                }
                
                .feature-card p {
                    color: var(--md-sys-color-on-surface-variant);
                    margin: 0;
                    line-height: 1.5;
                }
                
                .empty-state {
                    text-align: center;
                    padding: var(--spacing-xl);
                    color: var(--md-sys-color-on-surface-variant);
                }
                
                @media (max-width: 768px) {
                    .page-header h1 {
                        font-size: 2rem;
                    }
                    
                    .page-header p {
                        font-size: 1rem;
                    }
                    
                    .stats-grid {
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: var(--spacing-md);
                    }
                    
                    .actions-grid {
                        grid-template-columns: 1fr;
                        gap: var(--spacing-md);
                    }
                    
                    .features-grid {
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: var(--spacing-md);
                    }
                }
            </style>
        `;
    }
    
    async mount() {
        super.mount();
        this.setupActionCards();
    }
    
    setupActionCards() {
        const actionCards = this.$$('.action-card');
        
        actionCards.forEach(card => {
            card.addEventListener('click', () => {
                const action = card.getAttribute('data-action');
                this.handleAction(action);
            });
        });
        
        // Configurar eventos para itens recentes
        const recentItems = this.$$('.recent-item');
        recentItems.forEach(item => {
            item.addEventListener('click', () => {
                const formId = item.getAttribute('data-form-id');
                const action = item.getAttribute('data-action');
                
                if (formId && action) {
                    this.router.navigate(`${action}/${formId}`);
                }
            });
        });
    }
    
    handleAction(action) {
        switch (action) {
            case 'new-form':
                this.router.navigate('new-form');
                break;
            case 'view-forms':
                this.router.navigate('forms');
                break;
            case 'view-responses':
                this.router.navigate('responses');
                break;
            case 'import-form':
                this.showImportDialog();
                break;
            default:
                console.warn('Ação não reconhecida:', action);
        }
    }
    
    calculateStats() {
        const forms = window.KyrionForms.forms || [];
        const responses = window.KyrionForms.responses || [];
        
        this.stats.totalForms = forms.length;
        this.stats.totalResponses = responses.length;
        
        // Formulários recentes (últimos 5)
        this.stats.recentForms = forms
            .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
            .slice(0, 5);
            
        // Respostas recentes (últimas 5)
        this.stats.recentResponses = responses
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
            .slice(0, 5);
    }
    
    getAverageResponses() {
        const forms = window.KyrionForms.forms || [];
        const responses = window.KyrionForms.responses || [];
        
        if (forms.length === 0) return '0';
        
        const average = responses.length / forms.length;
        return Math.round(average).toString();
    }
    
    getRecentFormsCount() {
        const forms = window.KyrionForms.forms || [];
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
        
        const recentCount = forms.filter(form => 
            new Date(form.createdAt) > oneWeekAgo
        ).length;
        
        return recentCount.toString();
    }
    
    renderRecentForms() {
        if (this.stats.recentForms.length === 0) {
            return `
                <div class="recent-section">
                    <h2>Formulários Recentes</h2>
                    <div class="empty-state">
                        <p>Nenhum formulário criado ainda.</p>
                        <md-text-button onclick="window.app.router.navigate('new-form')">
                            Criar primeiro formulário
                        </md-text-button>
                    </div>
                </div>
            `;
        }
        
        const formsHtml = this.stats.recentForms.map(form => `
            <div class="recent-item" data-form-id="${form.id}" data-action="edit-form">
                <div class="recent-item-content">
                    <h4 class="recent-item-title">${form.title || 'Formulário sem título'}</h4>
                    <p class="recent-item-meta">
                        Atualizado em ${this.formatDate(form.updatedAt)} • 
                        ${form.questions?.length || 0} questões
                    </p>
                </div>
                <md-icon>chevron_right</md-icon>
            </div>
        `).join('');
        
        return `
            <div class="recent-section">
                <h2>Formulários Recentes</h2>
                <div class="recent-list">
                    ${formsHtml}
                </div>
            </div>
        `;
    }
    
    renderRecentResponses() {
        if (this.stats.recentResponses.length === 0) {
            return '';
        }
        
        const responsesHtml = this.stats.recentResponses.map(response => {
            const form = this.getForm(response.formId);
            const formTitle = form?.title || 'Formulário removido';
            
            return `
                <div class="recent-item" data-form-id="${response.formId}" data-action="responses">
                    <div class="recent-item-content">
                        <h4 class="recent-item-title">Resposta para: ${formTitle}</h4>
                        <p class="recent-item-meta">
                            Recebida em ${this.formatDate(response.createdAt)}
                        </p>
                    </div>
                    <md-icon>chevron_right</md-icon>
                </div>
            `;
        }).join('');
        
        return `
            <div class="recent-section">
                <h2>Respostas Recentes</h2>
                <div class="recent-list">
                    ${responsesHtml}
                </div>
            </div>
        `;
    }
    
    showImportDialog() {
        // TODO: Implementar dialog de importação
        this.showError('Funcionalidade de importação será implementada em breve');
    }
}
