import { TokenType } from "./lexer.js";
import { createNode } from "./parser.js";
import { prettify } from "./printer.js";
import { simplify, simplifyFNC, simplifyFND } from "./simplifier.js";

// Funções auxiliares para criar nós AST
export function AND(children) {
  return createNode(TokenType.CONJUNCTION, { children });
}

export function OR(children) {
  return createNode(TokenType.DISJUNCTION, { children });
}

export function NEG(child) {
  return createNode(TokenType.NEGATION, { child });
}

export function TRUE() {
  return createNode(TokenType.TRUE);
}

export function FALSE() {
  return createNode(TokenType.FALSE);
}

// Transformação principal - executa todas as etapas de transformação lógica
export function transformFormula(ast) {
  const transformationSteps = [];

  // Passo 1: Eliminar Implicações
  const withoutImplications = eliminateImplications(ast);
  transformationSteps.push({
    name: "Eliminar Implicações e Bicondicionais",
    result: prettify(withoutImplications),
  });

  // Passo 2: Mover Negações para Dentro (Forma Normal Negativa)
  const negativeNormalForm = moveNegationsInward(withoutImplications);
  transformationSteps.push({
    name: "Forma Normal Negativa",
    result: prettify(negativeNormalForm),
  });

  // Passo 3: Padronizar Variáveis
  const standardizedVariables = standardizeVariables(negativeNormalForm);
  transformationSteps.push({
    name: "Padronizar Variáveis",
    result: prettify(standardizedVariables),
  });

  // Passo 4: Forma Normal Prenex
  const prenexForm = convertToPrenexForm(standardizedVariables);
  transformationSteps.push({
    name: "Forma Normal Prenex",
    result: prettify(prenexForm),
  });

  // Passo 5: Forma Normal Conjuntiva Prenex
  const prenexCNF = convertToCNF(prenexForm);
  transformationSteps.push({
    name: "Forma Normal Conjuntiva Prenex",
    result: prettify(prenexCNF),
  });

  // Passo 6: Forma Normal Disjuntiva Prenex
  const prenexDNF = convertToDNF(prenexForm);
  transformationSteps.push({
    name: "Forma Normal Disjuntiva Prenex",
    result: prettify(prenexDNF),
  });

  // Passo 7: Skolemização
  const skolemizedForm = skolemization(prenexForm);
  transformationSteps.push({
    name: "Forma Normal Skolem",
    result: prettify(skolemizedForm),
  });

  // Passo 8: Eliminar Quantificadores Universais
  const withoutUniversalQuantifiers =
    eliminateUniversalQuantifiers(skolemizedForm);
  transformationSteps.push({
    name: "Eliminar Quantificadores Universais",
    result: prettify(withoutUniversalQuantifiers),
  });

  // Passo 9: Forma Normal Conjuntiva
  const CNF = convertToCNF(withoutUniversalQuantifiers);
  transformationSteps.push({
    name: "Forma Normal Conjuntiva",
    result: prettify(CNF),
  });

  // Passo 10: Forma Normal Disjuntiva
  const DNF = convertToDNF(withoutUniversalQuantifiers);
  transformationSteps.push({
    name: "Forma Normal Disjuntiva",
    result: prettify(DNF),
  });

  // Passo 11: Obter Forma Clausal
  const clausalForm = getClausalForm(convertToCNF(withoutUniversalQuantifiers));
  transformationSteps.push({
    name: "Forma Clausal",
    result: clausalForm,
  });

  // Passo 12: Identificar Cláusulas de Horn
  const hornClauses = getHornClauses(convertToCNF(withoutUniversalQuantifiers));
  transformationSteps.push({
    name: "Cláusulas de Horn",
    result: hornClauses,
  });

  return transformationSteps;
}

// Implementação das funções de transformação

// Elimina implicações e bicondicionais usando equivalências lógicas
function eliminateImplications(ast) {
  if (!ast) return ast;

  switch (ast.type) {
    case TokenType.IMPLIES:
      // A → B ≡ ¬A ∨ B
      return OR([
        NEG(eliminateImplications(ast.left)),
        eliminateImplications(ast.right),
      ]);

    case TokenType.BICONDITIONAL:
      // A ↔ B ≡ (A → B) ∧ (B → A)
      return AND([
        OR([
          NEG(eliminateImplications(ast.left)),
          eliminateImplications(ast.right),
        ]),
        OR([
          NEG(eliminateImplications(ast.right)),
          eliminateImplications(ast.left),
        ]),
      ]);

    case TokenType.NEGATION:
      return NEG(eliminateImplications(ast.child));

    case TokenType.CONJUNCTION:
    case TokenType.DISJUNCTION:
      const children = ast.children
        ? ast.children.map(eliminateImplications)
        : [eliminateImplications(ast.left), eliminateImplications(ast.right)];
      return { ...ast, children };

    case TokenType.FORALL:
    case TokenType.EXISTS:
      return {
        ...ast,
        child: eliminateImplications(ast.child),
      };

    default:
      return ast;
  }
}

// Aplica leis de De Morgan para mover negações para dentro
function moveNegationsInward(ast) {
  if (!ast) return ast;

  switch (ast.type) {
    case TokenType.NEGATION:
      const childNode = ast.child;

      // Dupla negação
      if (childNode.type === TokenType.NEGATION) {
        return moveNegationsInward(childNode.child);
      }

      // Leis de De Morgan para conjunções
      if (childNode.type === TokenType.CONJUNCTION) {
        const children = childNode.children
          ? childNode.children.map((child) => moveNegationsInward(NEG(child)))
          : [
              moveNegationsInward(NEG(childNode.left)),
              moveNegationsInward(NEG(childNode.right)),
            ];
        return OR(children);
      }

      // Leis de De Morgan para disjunções
      if (childNode.type === TokenType.DISJUNCTION) {
        const children = childNode.children
          ? childNode.children.map((child) => moveNegationsInward(NEG(child)))
          : [
              moveNegationsInward(NEG(childNode.left)),
              moveNegationsInward(NEG(childNode.right)),
            ];
        return AND(children);
      }

      // Negação de quantificadores
      if (childNode.type === TokenType.FORALL) {
        return {
          type: TokenType.EXISTS,
          variable: childNode.variable,
          child: moveNegationsInward(NEG(childNode.child)),
        };
      }
      if (childNode.type === TokenType.EXISTS) {
        return {
          type: TokenType.FORALL,
          variable: childNode.variable,
          child: moveNegationsInward(NEG(childNode.child)),
        };
      }

      // Caso base: fórmula atômica
      return NEG(moveNegationsInward(childNode));

    case TokenType.CONJUNCTION:
    case TokenType.DISJUNCTION:
      const children = ast.children
        ? ast.children.map(moveNegationsInward)
        : [moveNegationsInward(ast.left), moveNegationsInward(ast.right)];
      return { ...ast, children };

    case TokenType.FORALL:
    case TokenType.EXISTS:
      return {
        ...ast,
        child: moveNegationsInward(ast.child),
      };

    default:
      return ast;
  }
}

// Sequência padrão de nomes de variáveis para padronização
const variableSequence = ["x", "y", "z", "u", "v", "w", "\\alpha", "\\beta"];

// Padroniza nomes de variáveis para evitar conflitos
function standardizeVariables(
  ast,
  variableMap = new Map(),
  counter = { value: 0 }
) {
  if (!ast) return ast;

  switch (ast.type) {
    case TokenType.VAR:
      if (variableMap.has(ast.name)) {
        return { ...ast, name: variableMap.get(ast.name) };
      }
      return ast;

    case TokenType.PREDICATE:
      const newArguments = ast.args.map((argument) => {
        if (variableMap.has(argument)) {
          return variableMap.get(argument);
        }
        return argument;
      });
      return { ...ast, args: newArguments };

    case TokenType.FORALL:
    case TokenType.EXISTS:
      // Cria novo mapa para este escopo para não poluir escopo pai
      const newVariableMap = new Map(variableMap);
      const originalVariable = ast.variable;

      // Sempre cria novo nome de variável para este quantificador
      let newVariable;
      if (counter.value < variableSequence.length) {
        newVariable = variableSequence[counter.value];
      } else {
        newVariable = `x_${counter.value - variableSequence.length}`;
      }
      counter.value++;

      newVariableMap.set(originalVariable, newVariable);

      return {
        ...ast,
        variable: newVariable,
        child: standardizeVariables(ast.child, newVariableMap, counter),
      };

    case TokenType.NEGATION:
      return NEG(standardizeVariables(ast.child, variableMap, counter));

    case TokenType.CONJUNCTION:
    case TokenType.DISJUNCTION:
      const children = ast.children
        ? ast.children.map((child) =>
            standardizeVariables(child, variableMap, counter)
          )
        : [
            standardizeVariables(ast.left, variableMap, counter),
            standardizeVariables(ast.right, variableMap, counter),
          ];
      return { ...ast, children };

    default:
      return ast;
  }
}

// Converte para Forma Normal Prenex (todos quantificadores no início)
function convertToPrenexForm(ast) {
  if (!ast) return ast;

  // Função para extrair quantificadores recursivamente
  function extractQuantifiers(node) {
    if (node.type === TokenType.FORALL || node.type === TokenType.EXISTS) {
      const { quantifiers: childQuantifiers, matrix: childMatrix } =
        extractQuantifiers(node.child);
      return {
        quantifiers: [
          { type: node.type, variable: node.variable },
          ...childQuantifiers,
        ],
        matrix: childMatrix,
      };
    }

    if (node.type === TokenType.NEGATION) {
      const { quantifiers, matrix } = extractQuantifiers(node.child);
      return { quantifiers, matrix: NEG(matrix) };
    }

    if (
      node.type === TokenType.CONJUNCTION ||
      node.type === TokenType.DISJUNCTION
    ) {
      const children = node.children || [node.left, node.right];
      let allQuantifiers = [];
      let newChildren = [];

      for (const child of children) {
        const { quantifiers: childQuantifiers, matrix: childMatrix } =
          extractQuantifiers(child);
        allQuantifiers = [...allQuantifiers, ...childQuantifiers];
        newChildren.push(childMatrix);
      }

      return {
        quantifiers: allQuantifiers,
        matrix: { ...node, children: newChildren },
      };
    }

    // Para outros tipos, não há quantificadores
    return { quantifiers: [], matrix: node };
  }

  // Aplica quantificadores à matriz
  function applyQuantifiers(matrix, quantifiers) {
    if (quantifiers.length === 0) return matrix;

    const [firstQuantifier, ...remainingQuantifiers] = quantifiers;
    const childNode = applyQuantifiers(matrix, remainingQuantifiers);

    return {
      type: firstQuantifier.type,
      variable: firstQuantifier.variable,
      child: childNode,
    };
  }

  const { matrix, quantifiers } = extractQuantifiers(ast);

  // Mantém a ordem dos quantificadores (importante para semântica)
  return applyQuantifiers(matrix, quantifiers);
}

// Sequência de nomes de função para padronização
const functionSequence = ["f", "g", "h", "\\phi", "\\gamma", "\\chi", "\\zeta"];

// Skolemização: remove quantificadores existenciais usando funções de Skolem
function skolemization(ast) {
  if (!ast) return ast;

  let skolemFuncCounter = { value: 0 };
  let skolemConstCounter = { value: 1 };

  function skolemizeRecursively(
    node,
    universalQuantifiers = [],
    skolemFunctions = new Map()
  ) {
    switch (node.type) {
      case TokenType.EXISTS:
        // Cria função de Skolem para variável existencial
        const functionName =
          universalQuantifiers.length > 0
            ? skolemFuncCounter.value < functionSequence.length
              ? functionSequence[skolemFuncCounter.value++]
              : `f_${skolemFuncCounter.value++}`
            : `c_${skolemConstCounter.value++}`;
        const skolemArguments = universalQuantifiers.map(
          (quantifier) => quantifier.variable
        );

        // Substitui variável existencial por função de Skolem
        function substituteVariable(subNode) {
          if (!subNode) return subNode;

          if (
            subNode.type === TokenType.VAR &&
            subNode.name === node.variable
          ) {
            if (universalQuantifiers.length === 0) {
              return {
                type: TokenType.VAR,
                name: functionName,
              };
            } else {
              return {
                type: TokenType.PREDICATE,
                name: functionName,
                args: skolemArguments,
              };
            }
          }

          if (subNode.type === TokenType.PREDICATE) {
            return {
              ...subNode,
              args: subNode.args.map((argument) =>
                argument === node.variable
                  ? universalQuantifiers.length > 0
                    ? `${functionName}(${skolemArguments.join(",")})`
                    : functionName
                  : argument
              ),
            };
          }

          if (subNode.type === TokenType.NEGATION) {
            return NEG(substituteVariable(subNode.child));
          }

          if (
            subNode.type === TokenType.CONJUNCTION ||
            subNode.type === TokenType.DISJUNCTION
          ) {
            const children = subNode.children
              ? subNode.children.map(substituteVariable)
              : [
                  substituteVariable(subNode.left),
                  substituteVariable(subNode.right),
                ];
            return { ...subNode, children };
          }

          if (
            subNode.type === TokenType.FORALL ||
            subNode.type === TokenType.EXISTS
          ) {
            return {
              ...subNode,
              child: substituteVariable(subNode.child),
            };
          }

          return subNode;
        }

        return skolemizeRecursively(
          substituteVariable(node.child),
          universalQuantifiers,
          skolemFunctions
        );

      case TokenType.FORALL:
        return {
          ...node,
          child: skolemizeRecursively(
            node.child,
            [...universalQuantifiers, node],
            skolemFunctions
          ),
        };

      case TokenType.NEGATION:
        return NEG(
          skolemizeRecursively(
            node.child,
            universalQuantifiers,
            skolemFunctions
          )
        );

      case TokenType.CONJUNCTION:
      case TokenType.DISJUNCTION:
        const children = node.children
          ? node.children.map((child) =>
              skolemizeRecursively(child, universalQuantifiers, skolemFunctions)
            )
          : [
              skolemizeRecursively(
                node.left,
                universalQuantifiers,
                skolemFunctions
              ),
              skolemizeRecursively(
                node.right,
                universalQuantifiers,
                skolemFunctions
              ),
            ];
        return { ...node, children };

      default:
        return node;
    }
  }

  return skolemizeRecursively(ast);
}

// Elimina quantificadores universais (após skolemização)
function eliminateUniversalQuantifiers(ast) {
  if (!ast) return ast;

  switch (ast.type) {
    case TokenType.FORALL:
      return eliminateUniversalQuantifiers(ast.child);

    case TokenType.NEGATION:
      return NEG(eliminateUniversalQuantifiers(ast.child));

    case TokenType.CONJUNCTION:
    case TokenType.DISJUNCTION:
      const children = ast.children
        ? ast.children.map(eliminateUniversalQuantifiers)
        : [
            eliminateUniversalQuantifiers(ast.left),
            eliminateUniversalQuantifiers(ast.right),
          ];
      return { ...ast, children };

    default:
      return ast;
  }
}

// Normaliza children para estrutura plana
export function normalizeChildren(children) {
  const flattenedChildren = [];
  for (const child of children) {
    if (!child) continue;
    if (Array.isArray(child))
      flattenedChildren.push(...normalizeChildren(child));
    else if (
      child.type === TokenType.CONJUNCTION ||
      child.type === TokenType.DISJUNCTION
    ) {
      // Achata operações aninhadas do mesmo tipo
      if (child.children) {
        flattenedChildren.push(...normalizeChildren(child.children));
      } else if (child.left && child.right) {
        flattenedChildren.push(...normalizeChildren([child.left, child.right]));
      } else {
        flattenedChildren.push(child);
      }
    } else flattenedChildren.push(child);
  }
  return flattenedChildren;
}

// Distribui disjunções sobre conjunções para FNC
function distributeCNF(node) {
  if (!node) return null;

  // Manipula casos base e quantificadores primeiro
  if ([TokenType.FORALL, TokenType.EXISTS].includes(node.type)) {
    return createNode(node.type, {
      variable: node.variable,
      child: distributeCNF(node.child),
    });
  }

  if (
    node.type === TokenType.VAR ||
    node.type === TokenType.PREDICATE ||
    node.type === TokenType.TRUE ||
    node.type === TokenType.FALSE ||
    node.type === TokenType.NEGATION
  ) {
    return node;
  }

  // Distribui children recursivamente
  const children = extractAllChildren(node).map(distributeCNF);

  if (node.type === TokenType.DISJUNCTION) {
    // Procura por conjunções para distribuir
    const conjunctions = [];
    const nonConjunctions = [];

    for (const child of children) {
      if (child.type === TokenType.CONJUNCTION) {
        conjunctions.push(child);
      } else {
        nonConjunctions.push(child);
      }
    }

    if (conjunctions.length === 0) {
      return OR(children);
    }

    // Distribui: (A ∧ B) ∨ C → (A ∨ C) ∧ (B ∨ C)
    const firstConjunction = conjunctions[0];
    const remainingChildren = [...nonConjunctions, ...conjunctions.slice(1)];

    const distributedChildren = firstConjunction.children.map(
      (conjunctionChild) =>
        distributeCNF(OR([conjunctionChild, ...remainingChildren]))
    );

    return AND(distributedChildren);
  }

  if (node.type === TokenType.CONJUNCTION) {
    return AND(children);
  }

  return node;
}

// Distribui conjunções sobre disjunções para FND
function distributeDNF(node) {
  if (!node) return null;

  // Manipula casos base e quantificadores primeiro
  if ([TokenType.FORALL, TokenType.EXISTS].includes(node.type)) {
    return createNode(node.type, {
      variable: node.variable,
      child: distributeDNF(node.child),
    });
  }

  if (
    node.type === TokenType.VAR ||
    node.type === TokenType.PREDICATE ||
    node.type === TokenType.TRUE ||
    node.type === TokenType.FALSE ||
    node.type === TokenType.NEGATION
  ) {
    return node;
  }

  // Distribui children recursivamente
  const children = extractAllChildren(node).map(distributeDNF);

  if (node.type === TokenType.CONJUNCTION) {
    // Procura por disjunções para distribuir
    const disjunctions = [];
    const nonDisjunctions = [];

    for (const child of children) {
      if (child.type === TokenType.DISJUNCTION) {
        disjunctions.push(child);
      } else {
        nonDisjunctions.push(child);
      }
    }

    if (disjunctions.length === 0) {
      return AND(children);
    }

    // Distribui: (A ∨ B) ∧ C → (A ∧ C) ∨ (B ∧ C)
    const firstDisjunction = disjunctions[0];
    const remainingChildren = [...nonDisjunctions, ...disjunctions.slice(1)];

    const distributedChildren = firstDisjunction.children.map(
      (disjunctionChild) =>
        distributeDNF(AND([disjunctionChild, ...remainingChildren]))
    );

    return OR(distributedChildren);
  }

  if (node.type === TokenType.DISJUNCTION) {
    return OR(children);
  }

  return node;
}

// Converte para Forma Normal Conjuntiva (CNF)
function convertToCNF(ast) {
  const distributedAST = distributeCNF(ast);
  return simplifyFNC(distributedAST);
}

// Converte para Forma Normal Disjuntiva (DNF)
function convertToDNF(ast) {
  const distributedAST = distributeDNF(ast);
  return simplifyFND(distributedAST);
}

// Extrai todos os children de um nó
function extractAllChildren(node) {
  if (node.children) return node.children;
  if (node.left && node.right) return [node.left, node.right];
  return [node];
}

// Normaliza árvore binária para n-ária
function normalizeBinaryToNaryTree(ast) {
  if (!ast) return ast;

  if (
    ast.type === TokenType.CONJUNCTION ||
    ast.type === TokenType.DISJUNCTION
  ) {
    const children = [];

    function collectChildren(node) {
      if (node.type === ast.type) {
        node.children.forEach(collectChildren);
      } else {
        children.push(normalizeBinaryToNaryTree(node));
      }
    }

    collectChildren(ast);
    return { ...ast, children };
  }

  if (ast.type === TokenType.NEGATION) {
    return NEG(normalizeBinaryToNaryTree(ast.child));
  }

  if (ast.type === TokenType.FORALL || ast.type === TokenType.EXISTS) {
    return {
      ...ast,
      child: normalizeBinaryToNaryTree(ast.child),
    };
  }

  return ast;
}

// Obtém representação em forma clausal
function getClausalForm(ast) {
  if (!ast) return "";

  const normalizedTree = normalizeBinaryToNaryTree(ast);

  if (normalizedTree.type === TokenType.CONJUNCTION) {
    const clauses = normalizedTree.children.map(
      (clause, index) => `C_{${index + 1}}: ${prettify(clause)}`
    );

    return clauses.join("\\\\");
  }

  return `C_{1}: ${prettify(normalizedTree)}`;
}

// Obtém Cláusulas de Horn
function getHornClauses(ast) {
  if (!ast) return "\\texttt{Nenhuma Cláusula de Horn}";

  const normalizedTree = normalizeBinaryToNaryTree(ast);

  let clauses = [];

  if (normalizedTree.type === TokenType.CONJUNCTION) {
    clauses = normalizedTree.children.filter((child) => child);
  } else {
    clauses = [normalizedTree];
  }

  const hornClauses = [];

  clauses.forEach((clause, index) => {
    if (isHornClause(clause)) {
      const clauseType = getHornClauseType(clause);
      hornClauses.push(`C_{${index + 1}} (${clauseType}): ${prettify(clause)}`);
    }
  });

  if (hornClauses.length === 0) {
    return "\\texttt{Nenhuma Cláusula de Horn}";
  }

  return hornClauses.join("\\\\");
}

// Verifica se uma cláusula é uma Cláusula de Horn
function isHornClause(clause) {
  if (!clause) return false;

  // Extrai todos os literais da cláusula
  const literals = extractLiterals(clause);

  // Conta literais positivos
  let positiveCount = 0;

  for (const literal of literals) {
    if (isPositiveLiteral(literal)) {
      positiveCount++;
      // Se tem mais de 1 literal positivo, não é Horn
      if (positiveCount > 1) {
        return false;
      }
    }
  }

  return true;
}

// Determina o tipo da Cláusula de Horn
function getHornClauseType(clause) {
  const literals = extractLiterals(clause);
  let positiveCount = 0;
  let negativeCount = 0;
  let hasTrue = false;
  let hasFalse = false;

  for (const literal of literals) {
    if (literal.type === TokenType.TRUE) {
      hasTrue = true;
      positiveCount++;
    } else if (literal.type === TokenType.FALSE) {
      hasFalse = true;
      negativeCount++; // FALSE behaves like a negative constraint
    } else if (isPositiveLiteral(literal)) {
      positiveCount++;
    } else {
      negativeCount++;
    }
  }

  // Special case: clause contains FALSE
  if (hasFalse) {
    // If FALSE is the only literal, it's the empty clause
    if (literals.length === 1) return "Vazia";
    // Otherwise, it's a contradictory clause
    return "Inválida (contém contradição)";
  }

  // Special case: clause contains TRUE
  if (hasTrue) {
    // TRUE makes the whole clause tautological
    return "Tautologia";
  }

  // Standard Horn clause classification
  if (positiveCount === 1 && negativeCount === 0) {
    return "Fato";
  } else if (positiveCount === 1 && negativeCount > 0) {
    return "Regra";
  } else if (positiveCount === 0 && negativeCount > 0) {
    return "Consulta";
  } else if (positiveCount === 0 && negativeCount === 0) {
    return "Vazia";
  }

  return "Não é Horn (múltiplos positivos)";
}

// Extrai todos os literais de uma cláusula (disjunção)
function extractLiterals(node) {
  const literals = [];

  function extractRecursive(currentNode) {
    if (!currentNode) return;

    if (currentNode.type === TokenType.DISJUNCTION) {
      if (currentNode.children) {
        currentNode.children.forEach(extractRecursive);
      } else if (currentNode.left && currentNode.right) {
        extractRecursive(currentNode.left);
        extractRecursive(currentNode.right);
      }
    } else {
      literals.push(currentNode);
    }
  }

  extractRecursive(node);
  return literals;
}

// Verifica se um literal é positivo
function isPositiveLiteral(node) {
  if (!node) return false;

  if (node.type === TokenType.NEGATION) {
    return false;
  }

  // TRUE is always positive
  if (node.type === TokenType.TRUE) {
    return true;
  }

  // FALSE is never positive (it's a contradiction)
  if (node.type === TokenType.FALSE) {
    return false;
  }

  return node.type === TokenType.VAR || node.type === TokenType.PREDICATE;
}

// Função principal para executar todas as transformações
export function executeCompleteTransformations(ast) {
  return transformFormula(ast);
}
