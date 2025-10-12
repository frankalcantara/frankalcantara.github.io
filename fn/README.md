# Transformador de Fórmulas Lógicas de Primeira Ordem

Transformador interativo de fórmulas lógicas (FOL) que exibe passo-a-passo a conversão para formas normais (FNN/FNC/FND), skolemização, forma clausal e identificação de cláusulas de Horn. Interface web leve com KaTeX para renderização de LaTeX.

Visite o frontend: index.html  
Código de inicialização: index.js  
Estilos: index.css

Principais componentes (pipeline)
- Tokenização: `lexer.tokenize` — tokeniza a entrada LaTeX-like.
- Parsing: `parser.parse` — constrói a AST a partir dos tokens.
- Transformações completas: `transformer.executeCompleteTransformations` — orquestra todas as etapas de transformação.
- Impressão prettificada: `printer.prettify` — converte AST para LaTeX legível.
- Simplificações: `simplifier.simplify` — redução e normalização (FNC/FND helpers).

Visão geral das etapas aplicadas (implementadas em `transformer.transformFormula`):
1. Eliminar implicações / bicondicionais
2. Mover negações para dentro (Forma Normal Negativa)
3. Padronizar variáveis
4. Converter para forma prenex
5. Obter CNF prenex e DNF prenex
6. Skolemização
7. Remover quantificadores universais
8. Gerar CNF / DNF finais e forma clausal
9. Identificar cláusulas de Horn

Matematicamente:
- Forma Normal Conjuntiva (CNF): $$\text{CNF} \equiv \bigwedge_{i} \bigvee_{j} L_{ij}$$
- Forma Normal Disjuntiva (DNF): $$\text{DNF} \equiv \bigvee_{i} \bigwedge_{j} L_{ij}$$

Instalação e execução local
- Basta servir os arquivos estáticos. Exemplo com Python:
````sh
# Servir a pasta atual na porta 8000
python3 -m http.server 8000
# Abra http://localhost:8000/index.html
````
- Arquivo principal da UI: index.html — inclui KaTeX e carrega index.js.

Como usar
- Abra a página e escreva a fórmula em LaTeX-like (ex.: `\forall x \exists y (P(x) \land Q(x,y))`) no campo esquerdo.
- Clique em "Executar Transformações" para ver as etapas no painel da direita.
- Use os chips de exemplos para preencher rapidamente.
- Cada passo pode ser copiado via botão de cópia.

Erros e feedback
- Erros de lexing geram instância de `lexer.LexingError` com indicador de posição.
- Erros de parsing geram `parser.ParsingError` com mensagem e ponteiro para a coluna/linha.
- A interface mostra o indicador posicional e a mensagem para facilitar depuração.

Estrutura dos arquivos
- Frontend:
  - index.html
  - index.css
  - index.js
- Módulos principais:
  - lexer.js (tokenização)
  - parser.js (parsing / AST)
  - transformer.js (pipeline de transformações)
  - simplifier.js (simplificações FNC/FND)
  - printer.js (prettify/LaTeX)

Exemplo rápido (pseudocódigo de uso interno)
````js
// tokeniza -> parse -> transforma
import { tokenize } from "./modules/lexer.js";
import { parse } from "./modules/parser.js";
import { executeCompleteTransformations } from "./modules/transformer.js";

const consumer = tokenize("\\forall x P(x) \\to Q(x)");
const ast = parse(consumer);
const steps = executeCompleteTransformations(ast);
console.log(steps.map(s => s.name + ": " + s.result));
````

Contribuindo
- Abra uma issue descrevendo o problema/feature desejada.
- Envie PRs pequenas e focadas.
- Mantenha a consistência dos exports entre módulos e atualize teste manualmente via UI.

Boas práticas de extensão
- Para adicionar novos símbolos ou macros, atualize o mapa em `lexer.LiteralTokenMap` e ajuste o renderer de KaTeX em index.js.
- Para novas transformações, estenda `transformer.transformFormula` e exponha etapas no array retornado por `transformer.executeCompleteTransformations`.

Licença
- MIT

Contatos
- Desenvolvido por Renan da Silva Oliveira Andrade (@marshmll) — veja a referência no rodapé de index.html.

Obrigado por usar o Transformador de Fórmulas Lógicas.