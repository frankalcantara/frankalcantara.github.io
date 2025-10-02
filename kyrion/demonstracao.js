/**
 * Demonstração das funcionalidades implementadas no Módulo 1
 * Execute este arquivo no console do browser para testar
 */

// Dados de exemplo para demonstração
const exemploFormularios = [
    {
        id: 'form_1',
        title: 'Avaliação de Matemática',
        description: 'Teste sobre cálculo diferencial e integral',
        createdAt: new Date('2025-07-10'),
        updatedAt: new Date('2025-07-12'),
        questions: [
            { id: 'q1', type: 'multiple_choice', content: 'Qual é a derivada de x²?' },
            { id: 'q2', type: 'text', content: 'Explique o teorema fundamental do cálculo' }
        ]
    },
    {
        id: 'form_2',
        title: 'Quiz de Programação',
        description: 'Questões sobre JavaScript e algoritmos',
        createdAt: new Date('2025-07-08'),
        updatedAt: new Date('2025-07-11'),
        questions: [
            { id: 'q1', type: 'multiple_choice', content: 'O que é closure em JavaScript?' },
            { id: 'q2', type: 'text', content: 'Implemente um algoritmo de ordenação' }
        ]
    },
    {
        id: 'form_3',
        title: 'Física Quântica',
        description: 'Conceitos básicos de mecânica quântica',
        createdAt: new Date('2025-07-05'),
        updatedAt: new Date('2025-07-09'),
        questions: [
            { id: 'q1', type: 'multiple_choice', content: 'Qual é o princípio da incerteza de Heisenberg?' }
        ]
    }
];

const exemploRespostas = [
    {
        id: 'resp_1',
        formId: 'form_1',
        createdAt: new Date('2025-07-12'),
        answers: [
            { questionId: 'q1', value: '2x' },
            { questionId: 'q2', value: 'O teorema conecta derivação e integração...' }
        ]
    },
    {
        id: 'resp_2',
        formId: 'form_2',
        createdAt: new Date('2025-07-11'),
        answers: [
            { questionId: 'q1', value: 'Uma função que tem acesso ao escopo externo' }
        ]
    }
];

// Função para popular dados de exemplo
function popularDadosExemplo() {
    if (window.KyrionForms) {
        window.KyrionForms.forms = exemploFormularios;
        window.KyrionForms.responses = exemploRespostas;
        
        // Salvar no storage
        if (window.app && window.app.saveData) {
            window.app.saveData();
        }
        
        console.log('Dados de exemplo carregados');
        console.log('Formularios:', window.KyrionForms.forms.length);
        console.log('Respostas:', window.KyrionForms.responses.length);
        
        // Recarregar página home para mostrar dados
        if (window.app && window.app.router) {
            window.app.router.navigate('home');
        }
    } else {
        console.error('Kyrion Forms nao esta carregado');
    }
}

// Função para limpar dados
function limparDados() {
    if (window.KyrionForms) {
        window.KyrionForms.forms = [];
        window.KyrionForms.responses = [];
        
        // Limpar storage
        localStorage.removeItem('kyrion_forms');
        localStorage.removeItem('kyrion_responses');
        localStorage.removeItem('kyrion_settings');
        
        console.log('Dados limpos');
        
        // Recarregar página
        if (window.app && window.app.router) {
            window.app.router.navigate('home');
        }
    }
}

// Função para testar navegação
function testarNavegacao() {
    if (!window.app || !window.app.router) {
        console.error('Router nao esta disponivel');
        return;
    }
    
    const rotas = ['home', 'forms', 'new-form', 'responses'];
    let index = 0;
    
    console.log('Testando navegacao...');
    
    const interval = setInterval(() => {
        if (index >= rotas.length) {
            clearInterval(interval);
            console.log('Teste de navegacao concluido');
            window.app.router.navigate('home');
            return;
        }
        
        const rota = rotas[index];
        console.log(`Navegando para: ${rota}`);
        window.app.router.navigate(rota);
        index++;
    }, 2000);
}

// Função para testar responsividade
function testarResponsividade() {
    const tamanhos = [
        { nome: 'Mobile', largura: 375 },
        { nome: 'Tablet', largura: 768 },
        { nome: 'Desktop', largura: 1200 }
    ];
    
    let index = 0;
    
    console.log('Testando responsividade...');
    
    const interval = setInterval(() => {
        if (index >= tamanhos.length) {
            clearInterval(interval);
            console.log('Teste de responsividade concluido');
            // Restaurar tamanho original
            window.resizeTo(1200, 800);
            return;
        }
        
        const tamanho = tamanhos[index];
        console.log(`Testando ${tamanho.nome} (${tamanho.largura}px)`);
        window.resizeTo(tamanho.largura, 800);
        index++;
    }, 3000);
}

// Função para demonstrar recursos
function demonstrarRecursos() {
    console.log('KYRION FORMS - MODULO 1 COMPLETO');
    console.log('=====================================');
    console.log('');
    console.log('Recursos Implementados:');
    console.log('  • Layout responsivo com Material Design 3');
    console.log('  • Sistema de roteamento SPA');
    console.log('  • Navegação lateral adaptativa');
    console.log('  • Página inicial com dashboard');
    console.log('  • Armazenamento local (IndexedDB + localStorage)');
    console.log('  • Auto-save configurável');
    console.log('  • Tema claro/escuro automático');
    console.log('  • PWA ready (service worker pendente)');
    console.log('');
    console.log('Comandos disponiveis:');
    console.log('  popularDadosExemplo() - Carrega dados de teste');
    console.log('  limparDados() - Remove todos os dados');
    console.log('  testarNavegacao() - Testa navegação automática');
    console.log('  testarResponsividade() - Testa diferentes tamanhos');
    console.log('');
    console.log('Proximo: Modulo 2 - Sistema de Formularios');
    console.log('  • Criar/editar formulários');
    console.log('  • Gerenciar perguntas');
    console.log('  • Preview em tempo real');
    console.log('  • Autosave');
}

// Verificar se o script está sendo executado no browser
if (typeof window !== 'undefined') {
    // Aguardar carregamento da aplicação
    window.addEventListener('load', () => {
        setTimeout(() => {
            demonstrarRecursos();
            
            // Disponibilizar funções globalmente para teste
            window.demonstracao = {
                popularDadosExemplo,
                limparDados,
                testarNavegacao,
                testarResponsividade,
                demonstrarRecursos
            };
            
            console.log('Digite demonstracao.popularDadosExemplo() para carregar dados de teste');
        }, 2000);
    });
}

// Exportar para módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        exemploFormularios,
        exemploRespostas,
        popularDadosExemplo,
        limparDados,
        testarNavegacao,
        testarResponsividade,
        demonstrarRecursos
    };
}
