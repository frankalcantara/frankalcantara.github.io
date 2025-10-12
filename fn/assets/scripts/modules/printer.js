import { TokenType } from "./lexer.js";

// Função principal para formatar a AST em uma string bonita
export function prettify(node) {
  switch (node.type) {
    case TokenType.VAR:
      return node.name;

    case TokenType.PREDICATE:
      return `${node.name}(${node.args.join(", ")})`;

    case TokenType.NEGATION:
      return `\\neg ${wrapIfNeeded(node.child, node)}`;

    case TokenType.FORALL:
      return `\\forall ${node.variable} ${wrapIfNeeded(node.child, node)}`;

    case TokenType.EXISTS:
      return `\\exists ${node.variable} ${wrapIfNeeded(node.child, node)}`;

    case TokenType.CONJUNCTION:
      return node.children && node.children.length > 0
        ? node.children
            .map((child) => wrapIfNeeded(child, node))
            .join(" \\land ")
        : `${wrapIfNeeded(node.left, node)} \\land ${wrapIfNeeded(
            node.right,
            node
          )}`;

    case TokenType.DISJUNCTION:
      return node.children && node.children.length > 0
        ? node.children
            .map((child) => wrapIfNeeded(child, node))
            .join(" \\lor ")
        : `${wrapIfNeeded(node.left, node)} \\lor ${wrapIfNeeded(
            node.right,
            node
          )}`;

    case TokenType.IMPLIES:
      return `${wrapIfNeeded(node.left, node)} \\to ${wrapIfNeeded(
        node.right,
        node
      )}`;

    case TokenType.BICONDITIONAL:
      return `${wrapIfNeeded(node.left, node)} \\leftrightarrow ${wrapIfNeeded(
        node.right,
        node
      )}`;

    case TokenType.TRUE:
      return "\\top";

    case TokenType.FALSE:
      return "\\bot";

    default:
      throw new Error("Tipo de nó AST desconhecido: " + node.type);
  }
}

// Função auxiliar para adicionar parênteses quando necessário baseado na precedência
function wrapIfNeeded(childNode, parentNode) {
  if (!childNode) return "";

  // Tabela de precedência: valores mais altos = ligação mais forte
  const precedenceTable = {
    EXISTS: 7,
    FORALL: 7,
    PREDICATE: 6,
    VAR: 5,
    NEGATION: 5,
    CONJUNCTION: 4,
    DISJUNCTION: 3,
    IMPLICATION: 2,
    BICONDITIONAL: 1,
  };

  const childPrecedence = precedenceTable[childNode.type] || 0;
  const parentPrecedence = precedenceTable[parentNode.type] || 0;

  const childString = prettify(childNode);

  // Se a precedência do filho é estritamente menor, envolve com parênteses
  if (childPrecedence < parentPrecedence) {
    return `(${childString})`;
  }

  // Garante que (A ∧ B) dentro de ∨ seja envolvido com parênteses
  if (
    parentNode.type === TokenType.DISJUNCTION &&
    childNode.type === TokenType.CONJUNCTION
  ) {
    return `(${childString})`;
  }

  return childString;
}
