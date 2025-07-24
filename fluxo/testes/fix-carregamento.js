// Event listener SIMPLES para carregamento de exemplos - APENAS ESTA FUNÇÃO
document.addEventListener('DOMContentLoaded', function() {
    // Aguardar um pouco para garantir que tudo carregou
    setTimeout(function() {
        const exampleSelector = document.getElementById('example-selector');
        const editor = document.getElementById('mermaid-editor');
        
        console.log('🔍 Verificando elementos...');
        console.log('Selector encontrado:', !!exampleSelector);
        console.log('Editor encontrado:', !!editor);
        
        if (exampleSelector && editor) {
            console.log('✅ Configurando event listener SIMPLES...');
            
            exampleSelector.addEventListener('change', function() {
                const selectedValue = this.value;
                console.log('🎯 EXEMPLO SELECIONADO:', selectedValue);
                
                if (selectedValue) {
                    // Exemplos locais
                    const examples = {
                        basico: `flowchart TD
    A[Início] --> B[Ler nome]
    B --> C[Mostrar saudação]
    C --> D[Fim]`,
                        decisao: `flowchart TD
    A[Início] --> B[Ler idade]
    B --> C{Maior que 18?}
    C -->|Sim| D[Pode votar]
    C -->|Não| E[Não pode votar]
    D --> F[Fim]
    E --> F`,
                        calculadora: `flowchart TD
    A[Início] --> B[Ler primeiro número]
    B --> C[Ler segundo número]
    C --> D[Ler operação]
    D --> E{Operação é +?}
    E -->|Sim| F[Somar números]
    E -->|Não| G{Operação é -?}
    G -->|Sim| H[Subtrair números]
    G -->|Não| I[Operação inválida]
    F --> J[Mostrar resultado]
    H --> J
    I --> J
    J --> K[Fim]`
                    };
                    
                    const codigo = examples[selectedValue];
                    if (codigo) {
                        console.log('📝 CARREGANDO CÓDIGO NO EDITOR...');
                        editor.value = codigo;
                        console.log('✅ CÓDIGO CARREGADO COM SUCESSO!');
                        
                        // Resetar seleção após 2 segundos
                        setTimeout(() => {
                            this.value = '';
                        }, 2000);
                    } else {
                        console.error('❌ Código não encontrado para:', selectedValue);
                    }
                }
            });
            
            console.log('✅ Event listener configurado com sucesso!');
        } else {
            console.error('❌ Elementos não encontrados!');
        }
    }, 1000);
});
