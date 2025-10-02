# Questions Registry - Sistema de Gerenciamento de Questões (SISTEMA EXPANDIDO)

## ✅ SISTEMA ATIVO E EXPANDIDO - V2.0

O `questionsRegistry` foi **expandido e reativado** como controlador central específico por formulário. Combina **metadados no registry** com **conteúdo extraído do DOM** para máxima flexibilidade. **Inclui sistema de edição inline de enunciado** com redimensionamento automático via CSS.

## 📋 Visão Geral (Sistema V2.0)

O `questionsRegistry` é agora o **controlador central** específico por formulário que gerencia metadados das questões, combinado com extração de conteúdo do DOM para máxima flexibilidade.

## 🏗️ Estrutura de Dados Expandida

### Formato do Registry V2.0

```javascript
this.questionsRegistry = [
    {
        id: "1",                    // ID único da questão
        type: "multiple-choice",    // Tipo da questão
        position: 0,                // Posição no array (índice)
        order: 1,                   // Ordem da questão no formulário
        focus: true,                // Se está sendo editada/em foco
        enunciado: "",             // Texto principal da questão (pergunta)
        justificativa: "",          // Texto de justificativa da questão
        rubrica: "",               // Texto da rubrica de avaliação
        selecionado: false,        // Se a questão está selecionada na interface
        rubricaform: false,        // Se o formulário de rubrica está aberto
        domElement: HTMLElement     // Referência ao elemento DOM
    },
    {
        id: "q_1752808191111",      
        type: "multiple-choice",    
        position: 1,                
        order: 2,                   
        focus: false,
        enunciado: "Qual é o valor de x na equação 2x + 5 = 13?",
        justificativa: "Esta questão avalia conhecimento em álgebra básica",
        rubrica: "Critério 1: Corretude da resposta\nCritério 2: Processo de resolução",
        selecionado: true,
        rubricaform: false,               
        domElement: HTMLElement     
    }
    // ... mais questões
];
```

### Descrição dos Campos

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `id` | string | Identificador único da questão. Questão inicial usa "1", outras usam timestamp |
| `type` | string | Tipo da questão. Atualmente sempre "multiple-choice" |
| `position` | number | Posição da questão no array (0-based). Usado para inserção inteligente |
| `order` | number | Ordem da questão no formulário (1-based). Usado para numeração e sequência |
| `focus` | boolean | Indica se a questão está sendo editada ou em foco no momento |
| `enunciado` | string | **NOVO** - Texto principal da questão (pergunta que será respondida) |
| `justificativa` | string | **NOVO** - Texto explicativo sobre a questão (para professores/criadores) |
| `rubrica` | string | **NOVO** - Critérios de avaliação e pontuação da questão |
| `selecionado` | boolean | **NOVO** - Se a questão está visualmente selecionada na interface |
| `rubricaform` | boolean | **NOVO** - Se o formulário de edição de rubrica está aberto |
| `domElement` | HTMLElement | Referência direta ao elemento DOM da questão no container |

## 🔄 Como Funciona V2.0

### 1. Ciclo de Vida Específico por Formulário

#### 1.1 Inicialização (Automática)
```javascript
// Chamado automaticamente em loadNewForm()
initializeFormRegistry() {
    this.currentFormId = `form_${Date.now()}`;  // ID único do formulário
    this.questionsRegistry = [];                // Registry limpo
    console.log(`📋 Registry inicializado para formulário: ${this.currentFormId}`);
}
```

#### 1.2 Destruição (Automática)
```javascript
// Chamado automaticamente ao navegar para fora de new-form
destroyFormRegistry() {
    console.log(`🗑️ Destruindo registry do formulário: ${this.currentFormId}`);
    this.questionsRegistry = null;
    this.currentFormId = null;
}
```

### 2. Sistema de Edição de Enunciado

#### 2.1 Funcionalidades Implementadas

**Edição Inline Inteligente**:
```javascript
// Clicar no display ativa edição
editQuestionStatement(questionId) {
    // Transforma display em textarea editável (60px altura)
    // Foca e seleciona texto existente
    // Permite edição com teclas Esc/Enter
}

// Salvamento automático no registry
saveQuestionStatement(questionId) {
    // Salva texto no registry via setQuestionEnunciado()
    // Atualiza display automaticamente
    // CSS height:auto redimensiona conforme conteúdo
}
```

**Redimensionamento Automático via CSS**:
```css
.statement-display {
    height: auto;           /* Ajuste automático ao conteúdo */
    white-space: pre-wrap;  /* Preserva quebras de linha */
    word-wrap: break-word;  /* Quebra palavras longas */
}

.statement-input {
    min-height: 60px;       /* Altura fixa na edição */
    resize: vertical;       /* Permite redimensionamento manual */
}
```

#### 2.2 Estrutura HTML do Enunciado

```html
<div class="question-content">
    <div class="question-statement">
        <label class="statement-label">Enunciado da questão</label>
        <div class="statement-display" onclick="editQuestionStatement('${questionId}')">
            Clique para adicionar o enunciado da questão
        </div>
        <textarea class="statement-input hidden" 
                onblur="saveQuestionStatement('${questionId}')"
                onkeydown="handleStatementKeydown(event, '${questionId}')">
        </textarea>
    </div>
</div>
```

### 3. Registro de Questões

Quando uma nova questão é criada:

```javascript
// Em createMultipleChoiceQuestion()
this.questionsRegistry.splice(insertPosition, 0, {
    id: questionId,
    type: 'multiple-choice',
    position: insertPosition,
    domElement: null  // Preenchido após criar HTML
});
```

### 3. Posicionamento Inteligente

```javascript
// Em determineInsertPosition()
const selectedQuestion = document.querySelector('.question-block.selected');
const selectedId = selectedQuestion.dataset.questionId;
const selectedIndex = this.questionsRegistry.findIndex(q => q.id === selectedId);

// Nova questão vai para selectedIndex + 1
return selectedIndex + 1;
```

## 🎯 Casos de Uso

### Questão Inicial
```javascript
// Contexto: 'initial'
{
    id: "1",
    type: "multiple-choice", 
    position: 0,
    order: 1,
    focus: true,  // Primeira questão recebe foco automaticamente
    domElement: <div class="question-block">...</div>
}
```

### Questão via FAB
```javascript
// Contexto: 'fab' - inserida após questão selecionada
{
    id: "q_1752808191111",
    type: "multiple-choice",
    position: 1,  // Após questão com position 0
    order: 2,     // Segunda questão na sequência
    focus: true,  // Nova questão recebe foco
    domElement: <div class="question-block">...</div>
}
```

### Questão via Botão
```javascript
// Contexto: 'button' - inserida após questão selecionada
{
    id: "q_1752808192222", 
    type: "multiple-choice",
    position: 2,  // Após questão com position 1
    order: 3,     // Terceira questão na sequência
    focus: true,  // Nova questão recebe foco
    domElement: <div class="question-block">...</div>
}
```

## 🆕 Gerenciamento de Ordem e Foco

### Campo `order` - Numeração do Formulário

O campo `order` mantém a sequência lógica das questões no formulário:

```javascript
// Ao inserir nova questão na posição 1
questionsRegistry = [
    { id: "1", position: 0, order: 1, focus: false },
    { id: "q_123", position: 1, order: 2, focus: true },  // Nova questão
    { id: "q_456", position: 2, order: 3, focus: false }  // Reordenada
];
```

**Características:**
- **1-based**: Começa em 1 (não em 0)
- **Sequencial**: Sempre 1, 2, 3, 4...
- **Auto-atualizada**: Recalculada quando questões são inseridas/removidas
- **Para exibição**: Usada para mostrar "Questão 1", "Questão 2", etc.

### Campo `focus` - Controle de Edição

O campo `focus` indica qual questão está sendo editada:

```javascript
// Apenas uma questão pode ter focus = true
questionsRegistry = [
    { id: "1", focus: false },      // Não está sendo editada
    { id: "q_123", focus: true },   // Está sendo editada
    { id: "q_456", focus: false }   // Não está sendo editada
];
```

**Características:**
- **Exclusivo**: Apenas uma questão pode ter `focus: true`
- **Automático**: Nova questão sempre recebe foco
- **Sincronizado**: Reflete a classe CSS `.selected` no DOM

## 🔍 Algoritmo de Posicionamento

### Lógica de Inserção

1. **Questão Inicial** (`context: 'initial'`)
   - Sempre na `position: 0`
   - Primeira questão do formulário

2. **FAB/Botões** (`context: 'fab'` ou `context: 'button'`)
   - Encontra questão selecionada no DOM
   - Busca posição no registry: `findIndex(q => q.id === selectedId)`
   - Insere na posição: `selectedIndex + 1`

3. **Fallback**
   - Se nenhuma questão selecionada: `registry.length` (final)

### Exemplo de Fluxo

```javascript
// Estado inicial
questionsRegistry = [
    { id: "1", position: 0 }
]

// Usuário seleciona questão "1" e clica FAB
// Nova questão inserida na position 1
questionsRegistry = [
    { id: "1", position: 0 },
    { id: "q_123", position: 1 }
]

// Usuário seleciona questão "1" novamente e clica botão
// Nova questão inserida na position 1, outras movem para frente
questionsRegistry = [
    { id: "1", position: 0 },
    { id: "q_456", position: 1 },  // Nova questão
    { id: "q_123", position: 2 }   // Questão anterior movida
]
```

## 🛠️ Funções que Utilizam o Registry

### Principais Funções

#### `determineInsertPosition(context)`
- **Entrada**: Contexto da criação ('initial', 'fab', 'button')
- **Saída**: Posição onde inserir nova questão
- **Uso**: Calcula onde posicionar baseado na questão selecionada

#### `createQuestionHTML(questionData, insertPosition)`
- **Entrada**: Dados da questão e posição
- **Saída**: HTML inserido no DOM
- **Uso**: Atualiza `domElement` no registry após criar HTML

### Funções de Ordem e Foco

#### `calculateQuestionOrder(insertPosition)`
- **Entrada**: Posição de inserção
- **Saída**: Ordem da nova questão
- **Uso**: Calcula sequência lógica (1, 2, 3...)

#### `updateQuestionsOrder()`
- **Entrada**: Nenhuma
- **Saída**: Registry atualizado
- **Uso**: Recalcula `position` e `order` de todas as questões

#### `setQuestionFocus(questionId)`
- **Entrada**: ID da questão
- **Saída**: Foco atualizado no registry
- **Uso**: Define uma questão como ativa, remove foco das outras

### Funções de Array

#### `findIndex()` e `splice()`
- **Array.findIndex()**: Encontra questão por ID
- **Array.splice()**: Insere questão na posição correta

## 📊 Benefícios do Sistema

### 1. **Posicionamento Inteligente**
- Questões sempre inseridas na posição logicamente correta
- Baseado na questão atualmente selecionada pelo usuário

### 2. **Performance**
- Busca rápida por ID: O(n) linear
- Inserção eficiente com `splice()`
- Referência direta ao DOM evita queries repetidas

### 3. **Manutenibilidade**
- Estrutura clara e bem definida
- Separação entre dados e DOM
- Fácil debugging e inspeção

### 4. **Extensibilidade**
- Pronto para outros tipos de questão
- Campos adicionais podem ser facilmente adicionados
- Base sólida para funcionalidades futuras

## 🔮 Possíveis Extensões

### Campos Adicionais
```javascript
{
    id: "q_123",
    type: "multiple-choice",
    position: 1,
    domElement: HTMLElement,
    // Novos campos possíveis:
    created: Date,              // Timestamp de criação
    modified: Date,             // Última modificação
    parent: "section_1",        // Seção pai (se houver)
    tags: ["math", "basic"],    // Tags da questão
    difficulty: "medium"        // Nível de dificuldade
}
```

### Operações Avançadas
- **Reordenação**: Drag & drop com atualização automática das positions
- **Busca**: Filtros por tipo, tags, dificuldade
- **Agrupamento**: Questões organizadas em seções
- **Validação**: Verificação de integridade dos dados

---

**Nota**: O `questionsRegistry` é fundamental para o funcionamento correto do sistema de criação de questões, garantindo que todas as operações sejam realizadas de forma consistente e previsível.