import { TokenType } from "./lexer.js";
import { TRUE, FALSE, NEG, OR, AND } from "./transformer.js";

// Gera uma chave única para um nó baseada na sua estrutura JSON
const getNormalizedNodeKey = (node) => {
  if (node.type === TokenType.VAR) {
    return JSON.stringify({ type: node.type, name: node.name });
  }
  if (node.type === TokenType.PREDICATE) {
    return JSON.stringify({
      type: node.type,
      name: node.name,
      args: node.args,
    });
  }
  if (node.type === TokenType.TRUE) {
    return JSON.stringify({ type: TokenType.TRUE });
  }
  if (node.type === TokenType.FALSE) {
    return JSON.stringify({ type: TokenType.FALSE });
  }
  // For negation, we already handle in getLiteralKey, so we shouldn't get here
  return JSON.stringify(node);
};

// Verifica se um nó é um literal (VAR, TRUE/FALSE, PREDICATE ou negação de atômico)
function isLiteral(node) {
  if (!node) return false;
  if (node.type === TokenType.TRUE || node.type === TokenType.FALSE)
    return true;
  if (node.type === TokenType.VAR) return true;
  if (node.type === TokenType.PREDICATE) return true;
  if (node.type === TokenType.NEGATION) {
    const childNode = node.child;
    return (
      childNode.type === TokenType.VAR ||
      childNode.type === TokenType.TRUE ||
      childNode.type === TokenType.FALSE ||
      childNode.type === TokenType.PREDICATE
    );
  }
  return false;
}

// Gera chave para literal, tratando negações com prefixo "!"
function getLiteralKey(node) {
  if (node.type === TokenType.NEGATION)
    return "!" + getNormalizedNodeKey(node.child);
  return getNormalizedNodeKey(node);
}

// Verifica se dois literais são complementares (um é negação do outro)
function areComplementary(nodeA, nodeB) {
  const keyA = getLiteralKey(nodeA);
  const keyB = getLiteralKey(nodeB);

  return keyA === "!" + keyB || keyB === "!" + keyA;
}

// Achata nós aninhados do mesmo tipo (A ∧ (B ∧ C)) -> (A ∧ B ∧ C)
function flattenSameType(operatorType, childrenNodes) {
  const flattenedChildren = [];
  for (const child of childrenNodes) {
    if (child.type === operatorType && child.children)
      flattenedChildren.push(...child.children);
    else flattenedChildren.push(child);
  }
  return flattenedChildren;
}

// Remove nós duplicados estruturalmente de um array
function removeDuplicateNodes(nodes) {
  const seenKeys = new Set();
  const uniqueNodes = [];
  for (const node of nodes) {
    const nodeKey = getNormalizedNodeKey(node);
    if (!seenKeys.has(nodeKey)) {
      seenKeys.add(nodeKey);
      uniqueNodes.push(node);
    }
  }
  return uniqueNodes;
}

// Função principal de simplificação de expressões lógicas
export function simplify(node) {
  if (!node) return null;

  switch (node.type) {
    case TokenType.TRUE:
    case TokenType.FALSE:
    case TokenType.VAR:
    case TokenType.PREDICATE:
      return node;

    case TokenType.NEGATION: {
      const simplifiedChild = simplify(node.child);
      // ~~A => A (dupla negação)
      if (simplifiedChild.type === TokenType.NEGATION)
        return simplify(simplifiedChild.child);
      // ~TRUE = FALSE ; ~FALSE = TRUE
      if (simplifiedChild.type === TokenType.TRUE) return FALSE();
      if (simplifiedChild.type === TokenType.FALSE) return TRUE();
      return NEG(simplifiedChild);
    }

    case TokenType.CONJUNCTION: {
      // Simplifica os filhos primeiro
      let childNodes = (node.children ?? [node.left, node.right])
        .filter(Boolean)
        .map(simplify);

      // Achata e remove duplicatas
      childNodes = flattenSameType(TokenType.CONJUNCTION, childNodes);
      childNodes = removeDuplicateNodes(childNodes);

      // Identidade e dominação: A ∧ TRUE = A ; A ∧ FALSE = FALSE
      if (childNodes.some((child) => child.type === TokenType.FALSE))
        return FALSE();
      childNodes = childNodes.filter((child) => child.type !== TokenType.TRUE);

      // Caso especial: conjunção vazia = TRUE
      if (childNodes.length === 0) return TRUE();
      if (childNodes.length === 1) return childNodes[0];

      // Verifica contradições literais em conjunções puras
      const allLiterals = childNodes.every(isLiteral);
      if (allLiterals) {
        for (let i = 0; i < childNodes.length; i++) {
          for (let j = i + 1; j < childNodes.length; j++) {
            if (areComplementary(childNodes[i], childNodes[j])) return FALSE();
          }
        }
      }

      // Mantém como está se for um termo FND (∧ de literais)
      return AND(childNodes);
    }

    case TokenType.DISJUNCTION: {
      let childNodes = (node.children ?? [node.left, node.right])
        .filter(Boolean)
        .map(simplify);

      childNodes = flattenSameType(TokenType.DISJUNCTION, childNodes);
      childNodes = removeDuplicateNodes(childNodes);

      // Identidade e dominação: A ∨ FALSE = A ; A ∨ TRUE = TRUE
      if (childNodes.some((child) => child.type === TokenType.TRUE))
        return TRUE();
      childNodes = childNodes.filter((child) => child.type !== TokenType.FALSE);

      if (childNodes.length === 0) return FALSE();
      if (childNodes.length === 1) return childNodes[0];

      // Detecta tautologia em disjunções de literais: L ∨ ~L => TRUE
      const allLiterals = childNodes.every(isLiteral);
      if (allLiterals) {
        for (let i = 0; i < childNodes.length; i++) {
          for (let j = i + 1; j < childNodes.length; j++) {
            if (areComplementary(childNodes[i], childNodes[j])) return TRUE();
          }
        }
      }

      return OR(childNodes);
    }

    // Pass-through para quantificadores (mantém estrutura)
    case TokenType.FORALL:
    case TokenType.EXISTS:
      return { ...node, child: simplify(node.child) };

    // Implicações e bicondicionais deveriam ter sido removidas antes da simplificação
    case TokenType.IMPLIES:
    case TokenType.BICONDITIONAL:
      return {
        ...node,
        left: simplify(node.left),
        right: simplify(node.right),
      };

    default:
      throw new Error("Tipo de nó desconhecido em simplify: " + node.type);
  }
}

// Simplificação específica para Forma Normal Conjuntiva (FNC)
export function simplifyFNC(rootNode) {
  const simplifiedNode = simplify(rootNode);
  if (simplifiedNode.type !== TokenType.CONJUNCTION) return simplifiedNode; // já é uma cláusula única ou constante

  // Converte cada cláusula para um conjunto normalizado de literais
  const clauses = simplifiedNode.children
    .map(simplifyClauseForFNC)
    .filter((clause) => clause.kind !== TokenType.TRUE); // Remove tautologias

  // Se qualquer cláusula é FALSE => toda FNC é FALSE
  if (clauses.some((clause) => clause.kind === TokenType.FALSE)) return FALSE();

  // FNC vazia
  if (clauses.length === 0) return TRUE();

  // Filtra apenas cláusulas válidas com conjuntos
  const validClauses = clauses.filter(
    (clause) => clause.kind === "CLAUSE" && clause.literalSet
  );

  // Subsunção: remove cláusulas que são superconjuntos de outras
  const keptClauses = [];
  for (let i = 0; i < validClauses.length; i++) {
    let isSubsumed = false;
    for (let j = 0; j < validClauses.length; j++) {
      if (i === j) continue;

      // Se j ⊆ i, então i é superconjunto (remove i)
      if (isSubset(validClauses[j].literalSet, validClauses[i].literalSet)) {
        isSubsumed = true;
        break;
      }
    }
    if (!isSubsumed) keptClauses.push(validClauses[i]);
  }

  if (keptClauses.length === 0) return TRUE(); // todas foram TRUE ou subsumidas

  // Reconstrói a AST a partir dos conjuntos
  const rebuiltClauses = keptClauses.map((clause) =>
    createDisjunctionFromSet(clause.literalSet)
  );

  if (rebuiltClauses.length === 1) return rebuiltClauses[0];

  return AND(rebuiltClauses);
}

// Simplificação específica para Forma Normal Disjuntiva (FND)
export function simplifyFND(rootNode) {
  const simplifiedNode = simplify(rootNode);
  if (simplifiedNode.type !== TokenType.DISJUNCTION) return simplifiedNode; // já é um termo único ou constante

  // Converte cada termo para um conjunto normalizado de literais
  const terms = simplifiedNode.children.map(simplifyTermForFND).filter(Boolean); // remove termos FALSE

  // Filtra apenas termos válidos com conjuntos
  const validTerms = terms.filter(
    (term) => term.kind === "TERM" && term.literalSet
  );

  // Se qualquer termo é TRUE => toda FND é TRUE
  if (validTerms.some((term) => term.kind === TokenType.TRUE)) return TRUE();

  // Subsunção: em FND, remove superconjuntos (A ∨ (A ∧ B) = A)
  const keptTerms = [];
  for (let i = 0; i < validTerms.length; i++) {
    let isSubsumed = false;
    for (let j = 0; j < validTerms.length; j++) {
      if (i === j) continue;
      if (isSubset(validTerms[j].literalSet, validTerms[i].literalSet)) {
        // j ⊆ i => i é superconjunto => remove i
        isSubsumed = true;
        break;
      }
    }
    if (!isSubsumed) keptTerms.push(validTerms[i]);
  }

  if (keptTerms.length === 0) return FALSE(); // todos os termos FALSE ou subsumidos

  const rebuiltTerms = keptTerms.map((term) =>
    createConjunctionFromSet(term.literalSet)
  );
  if (rebuiltTerms.length === 1) return rebuiltTerms[0];
  return OR(rebuiltTerms);
}

// Simplifica uma cláusula individual para FNC
function simplifyClauseForFNC(node) {
  // Uma cláusula é uma DISJUNÇÃO de literais (ou um literal único)
  if (node.type === TokenType.TRUE) return { kind: TokenType.TRUE };
  if (node.type === TokenType.FALSE) return { kind: TokenType.FALSE };

  let literals = [];
  if (node.type === TokenType.DISJUNCTION && node.children)
    literals = node.children;
  else literals = [node];

  // Simplifica cada literal e filtra apenas literais válidos
  literals = literals.map(simplify).filter(isLiteral);

  // Se cláusula contém literais complementares => cláusula é TRUE
  for (let i = 0; i < literals.length; i++) {
    for (let j = i + 1; j < literals.length; j++) {
      if (areComplementary(literals[i], literals[j])) {
        return { kind: TokenType.TRUE }; // Tautologia
      }
    }
  }

  // Remove literais FALSE e deduplica
  literals = literals.filter((literal) => literal.type !== TokenType.FALSE);

  // Verifica se ficou vazia após filtro
  if (literals.length === 0) return { kind: TokenType.FALSE };

  const literalSet = new Set(literals.map(getLiteralKey));

  return { kind: "CLAUSE", literalSet };
}

// Simplifica um termo individual para FND
function simplifyTermForFND(node) {
  // Um termo é uma CONJUNÇÃO de literais (ou um literal único)
  if (node.type === TokenType.TRUE) return { kind: TokenType.TRUE };
  if (node.type === TokenType.FALSE) return { kind: TokenType.FALSE };

  let literals = [];
  if (node.type === TokenType.CONJUNCTION && node.children)
    literals = node.children;
  else literals = [node];

  literals = literals.map(simplify).filter(isLiteral);

  // Se termo contém literais complementares => FALSE
  for (let i = 0; i < literals.length; i++) {
    for (let j = i + 1; j < literals.length; j++) {
      if (areComplementary(literals[i], literals[j]))
        return { kind: TokenType.FALSE };
    }
  }

  // Remove literais TRUE e deduplica
  literals = literals.filter((literal) => literal.type !== TokenType.TRUE);

  // Verifica se ficou vazia após filtro
  if (literals.length === 0) return { kind: TokenType.FALSE };

  const literalSet = new Set(literals.map(getLiteralKey));

  return { kind: "TERM", literalSet };
}

// Verifica se setA é subconjunto de setB
function isSubset(setA, setB) {
  for (const element of setA) if (!setB.has(element)) return false;
  return true;
}

// Cria disjunção a partir de um conjunto de literais
function createDisjunctionFromSet(literalSet) {
  const nodes = [...literalSet].map(createNodeFromLiteralKey);
  return nodes.length === 1 ? nodes[0] : OR(nodes);
}

// Cria conjunção a partir de um conjunto de literais
function createConjunctionFromSet(literalSet) {
  const nodes = [...literalSet].map(createNodeFromLiteralKey);
  return nodes.length === 1 ? nodes[0] : AND(nodes);
}

// Reconstrói nó a partir da chave do literal
function createNodeFromLiteralKey(key) {
  if (key.startsWith("!")) return NEG(JSON.parse(key.slice(1)));
  return JSON.parse(key);
}
