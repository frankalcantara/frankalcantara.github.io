import { TokenConsumer, TokenType, LiteralTokenMap } from "./lexer.js";

export class ParsingError extends Error {
  constructor(message, position, input) {
    super(message);
    this.name = "ParsingError";
    this.position = position;
    this.input = input;

    // Generate error display with pointer to the error position
    this.displayMessage = this.generateDisplayMessage();
  }

  generateDisplayMessage() {
    if (this.position === undefined || this.position < 0 || !this.input) {
      return this.message;
    }

    const lines = this.input.split("\n");
    let currentPos = 0;
    let lineNum = 0;
    let columnNum = 0;

    // Find the line and column where the error occurred
    for (let i = 0; i < lines.length; i++) {
      if (currentPos + lines[i].length >= this.position) {
        lineNum = i + 1;
        columnNum = this.position - currentPos + 1;
        break;
      }
      currentPos += lines[i].length + 1; // +1 for the newline character
    }

    const errorLine = lines[lineNum - 1];
    const pointer = " ".repeat(columnNum - 1) + "^";

    return `${this.message}\n\nAt line ${lineNum}, column ${columnNum}:\n\n${errorLine}\n${pointer}`;
  }
}

// Função auxiliar para criar nós da AST
export function createNode(type, properties = {}) {
  return { type, ...properties };
}

// Função principal de parsing - converte tokens em AST
export function parse(consumer = new TokenConsumer([])) {
  const abstractSyntaxTree = parseExpression(consumer);

  // Verifica se há tokens restantes não processados
  const nextToken = consumer.peek();
  if (nextToken !== null) {
    if (
      nextToken.type === TokenType.RPAREN ||
      nextToken.type === TokenType.RBRACKET ||
      nextToken.type === TokenType.RBRACE
    ) {
      const symbol = Object.keys(LiteralTokenMap).find(
        (key) => LiteralTokenMap[key] === nextToken.type
      );

      throw new ParsingError(
        `Parêntese/colchete/chave de fechamento extra: "${symbol}"`,
        nextToken.position,
        consumer.originalInput
      );
    }

    throw new ParsingError(
      `Token inesperado: ${JSON.stringify(nextToken.literal)}`,
      nextToken.position,
      consumer.originalInput
    );
  }

  return abstractSyntaxTree;
}

// Inicia o parsing a partir do nível de expressão
function parseExpression(consumer) {
  return parseBiconditional(consumer);
}

// Parsing de bicondicionais (↔)
function parseBiconditional(consumer) {
  let leftNode = parseImplication(consumer);

  while (consumer.peek()?.type === TokenType.BICONDITIONAL) {
    consumer.next();
    const rightNode = parseImplication(consumer);
    leftNode = createNode(TokenType.BICONDITIONAL, {
      left: leftNode,
      right: rightNode,
    });
  }

  return leftNode;
}

// Parsing de implicações (→)
function parseImplication(consumer) {
  let leftNode = parseDisjunction(consumer);

  while (consumer.peek()?.type === TokenType.IMPLIES) {
    consumer.next();
    const rightNode = parseDisjunction(consumer);
    leftNode = createNode(TokenType.IMPLIES, {
      left: leftNode,
      right: rightNode,
    });
  }

  return leftNode;
}

// Parsing de disjunções (∨)
function parseDisjunction(consumer) {
  let leftNode = parseConjunction(consumer);

  while (consumer.peek()?.type === TokenType.DISJUNCTION) {
    consumer.next();
    const rightNode = parseConjunction(consumer);
    leftNode = createNode(TokenType.DISJUNCTION, {
      left: leftNode,
      right: rightNode,
    });
  }

  return leftNode;
}

// Parsing de conjunções (∧)
function parseConjunction(consumer) {
  let leftNode = parseUnary(consumer);

  while (consumer.peek()?.type === TokenType.CONJUNCTION) {
    consumer.next();
    const rightNode = parseDisjunction(consumer);
    leftNode = createNode(TokenType.CONJUNCTION, {
      left: leftNode,
      right: rightNode,
    });
  }

  return leftNode;
}

// Parsing de operadores unários (¬)
function parseUnary(consumer) {
  if (consumer.peek()?.type === TokenType.NEGATION) {
    consumer.next();
    const childNode = parseUnary(consumer);
    return createNode(TokenType.NEGATION, { child: childNode });
  }

  return parseQuantifiers(consumer);
}

// Parsing de quantificadores (∀, ∃)
function parseQuantifiers(consumer) {
  const tokenType = consumer.peek()?.type;

  if ([TokenType.FORALL, TokenType.EXISTS].includes(tokenType)) {
    consumer.next(); // consome ∀ ou ∃

    // Espera uma variável logo após o quantificador
    const variableToken = consumer.peek();
    if (!variableToken || variableToken.type !== TokenType.VAR) {
      throw new ParsingError(
        `Esperava variável após quantificador, obteve ${
          variableToken?.literal || "nada"
        }. Verifique espaços.`,
        consumer.getCurrentPosition(),
        consumer.originalInput
      );
    }

    // Verifica se é uma variável minúscula
    if (variableToken.literal !== variableToken.literal.toLowerCase()) {
      throw new ParsingError(
        `Variáveis após quantificadores devem ser letras minúsculas (variável ${variableToken.literal})`,
        variableToken.position,
        consumer.originalInput
      );
    }

    consumer.next(); // consome a variável

    // Parseia o escopo da fórmula do quantificador
    const childNode = parseQuantifiers(consumer) || parsePrimary(consumer);

    return createNode(tokenType, {
      variable: variableToken.literal,
      child: childNode,
    });
  }

  return parsePrimary(consumer);
}

// Parsing de elementos primários (variáveis, predicados, constantes, parênteses)
function parsePrimary(consumer) {
  const currentToken = consumer.peek();

  if (!currentToken) {
    throw new ParsingError(
      "Fim de input inesperado",
      consumer.getCurrentPosition(),
      consumer.originalInput
    );
  }

  if (currentToken.type === TokenType.VAR) {
    consumer.next();
    return createNode(TokenType.VAR, {
      name: currentToken.literal,
      position: currentToken.position,
    });
  }

  if (currentToken.type === TokenType.PREDICATE) {
    const literal = currentToken.literal;

    // Extrai string dentro dos parênteses
    const regex = /\(([^)]+)\)/;
    const matchResult = literal.match(regex);

    if (!matchResult || !matchResult[1] || matchResult[1].length === 0) {
      throw new ParsingError(
        `Nenhum argumento pôde ser extraído do predicado: ${currentToken.literal}`,
        currentToken.position,
        consumer.originalInput
      );
    }

    // Divide argumentos: P(x, y) => ["x", "y"]
    const argumentsList = matchResult[1]
      .split(",")
      .map((argument) => argument.trim())
      .filter((argument) => {
        if (argument.length === 0) {
          throw new ParsingError(
            `Erro de sintaxe para o predicado ${literal}: ${argument}. Verifique vírgulas.`,
            currentToken.position,
            consumer.originalInput
          );
        }

        if (/^\\s*$/.test(argument)) return false;

        if (argument.length > 1) {
          throw new ParsingError(
            `Argumento inválido para o predicado ${literal}: ${argument}. Argumentos devem ser apenas letras únicas.`,
            currentToken.position,
            consumer.originalInput
          );
        }

        if (argument !== argument.toLowerCase()) {
          throw new ParsingError(
            `Argumento inválido para o predicado ${literal}: ${argument}. Argumentos devem ser letras minúsculas.`,
            currentToken.position,
            consumer.originalInput
          );
        }

        return true;
      });

    if (argumentsList.length === 0) {
      throw new ParsingError(
        `Predicado não forneceu argumentos: ${currentToken.literal}`,
        currentToken.position,
        consumer.originalInput
      );
    }

    consumer.next();
    return createNode(TokenType.PREDICATE, {
      name: literal[0],
      args: argumentsList,
      position: currentToken.position,
    });
  }

  if (
    currentToken.type === TokenType.TRUE ||
    currentToken.type === TokenType.FALSE
  ) {
    consumer.next();
    return createNode(currentToken.type, {
      position: currentToken.position,
    });
  }

  if (currentToken.type === TokenType.NEGATION) {
    consumer.next();
    const childNode = parseUnary(consumer);
    return createNode(TokenType.NEGATION, {
      child: childNode,
      position: currentToken.position,
    });
  }

  if (currentToken.type === TokenType.UNKNOWN) {
    throw new ParsingError(
      `Token desconhecido: ${currentToken.literal}`,
      currentToken.position,
      consumer.originalInput
    );
  }

  // Parsing de expressões entre parênteses/colchetes/chaves
  if (
    currentToken.type === TokenType.LPAREN ||
    currentToken.type === TokenType.LBRACE ||
    currentToken.type === TokenType.LBRACKET
  ) {
    const openingToken = consumer.next();

    const expressionNode = parseExpression(consumer);

    const closingToken = consumer.peek();

    const expectedClosingToken = {
      [TokenType.LPAREN]: TokenType.RPAREN,
      [TokenType.LBRACE]: TokenType.RBRACE,
      [TokenType.LBRACKET]: TokenType.RBRACKET,
    };

    if (
      !closingToken ||
      closingToken.type !== expectedClosingToken[openingToken.type]
    ) {
      const expectedType = expectedClosingToken[openingToken.type];
      const expectedSymbol = Object.keys(LiteralTokenMap).find(
        (key) => LiteralTokenMap[key] === expectedType
      );

      if (
        closingToken &&
        (closingToken.type === TokenType.RPAREN ||
          closingToken.type === TokenType.RBRACE ||
          closingToken.type === TokenType.RBRACKET)
      ) {
        const actualSymbol = Object.keys(LiteralTokenMap).find(
          (key) => LiteralTokenMap[key] === closingToken.type
        );

        throw new ParsingError(
          `Fechamento incorreto: esperado "${expectedSymbol}" mas encontrado "${actualSymbol}"`,
          closingToken.position,
          consumer.originalInput
        );
      }

      throw new ParsingError(
        `Parênteses/colchetes/chaves não foi fechado(a) (esperado: "${expectedSymbol}")`,
        consumer.getCurrentPosition(),
        consumer.originalInput
      );
    }

    consumer.next();
    return expressionNode;
  }

  throw new ParsingError(
    `Token inesperado: ${currentToken.literal}`,
    currentToken.position,
    consumer.originalInput
  );
}
