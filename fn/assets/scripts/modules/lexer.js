// Mapeamento de tipos de tokens para lógica proposicional e de predicados
export const TokenType = {
  UNKNOWN: "UNKNOWN",
  CONJUNCTION: "CONJUNCTION",
  DISJUNCTION: "DISJUNCTION",
  IMPLIES: "IMPLIES",
  BICONDITIONAL: "BICONDITIONAL",
  NEGATION: "NEGATION",
  LPAREN: "LPAREN",
  RPAREN: "RPAREN",
  LBRACKET: "LBRACKET",
  RBRACKET: "RBRACKET",
  LBRACE: "LBRACE",
  RBRACE: "RBRACE",
  TRUE: "TRUE",
  FALSE: "FALSE",
  VAR: "VAR",
  PREDICATE: "PREDICATE",
  FORALL: "FORALL",
  EXISTS: "EXISTS",
};

// Mapeamento de literais para seus respectivos tipos de token
export const LiteralTokenMap = {
  "\\land": TokenType.CONJUNCTION,
  "\\wedge": TokenType.CONJUNCTION,
  "\\lor": TokenType.DISJUNCTION,
  "\\vee": TokenType.DISJUNCTION,
  "\\rightarrow": TokenType.IMPLIES,
  "\\implies": TokenType.IMPLIES,
  "\\leftrightarrow": TokenType.BICONDITIONAL,
  "\\iff": TokenType.BICONDITIONAL,
  "\\neg": TokenType.NEGATION,
  "\\lnot": TokenType.NEGATION,
  "(": TokenType.LPAREN,
  ")": TokenType.RPAREN,
  "[": TokenType.LBRACKET,
  "]": TokenType.RBRACKET,
  "{": TokenType.LBRACE,
  "}": TokenType.RBRACE,
  "\\top": TokenType.TRUE,
  "\\bot": TokenType.FALSE,
  "\\forall": TokenType.FORALL,
  "\\exists": TokenType.EXISTS,
};

// Custom error class for lexing errors with position information
export class LexingError extends Error {
  constructor(message, position, input) {
    super(message);
    this.name = "LexingError";
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

// Consumidor de tokens para navegar pela lista de tokens
export class TokenConsumer {
  constructor(tokens, originalInput = "") {
    this.currentIndex = 0;
    this.tokens = tokens;
    this.originalInput = originalInput;
  }

  // Retorna o próximo token sem consumi-lo (lookahead)
  peek() {
    if (this.currentIndex >= this.tokens.length) return null;
    return this.tokens[this.currentIndex];
  }

  // Retorna o próximo token e avança o índice
  next() {
    if (this.currentIndex >= this.tokens.length) return null;
    return this.tokens[this.currentIndex++];
  }

  // Retorna a posição atual da cabeça de leitura.
  getCurrentPosition() {
    if (this.currentIndex >= this.tokens.length) {
      return this.originalInput.length;
    }
    return this.tokens[this.currentIndex].position;
  }
}

// Verifica se o buffer é prefixo de alguma chave no mapa de tokens
function isPrefixOfAnyKey(buffer) {
  return Object.keys(LiteralTokenMap).some((key) => key.startsWith(buffer));
}

// Função auxiliar para criar objetos token
function createToken(type, literal, position = -1) {
  return { type, literal, position };
}

// Função principal de tokenização - converte string em tokens
export function tokenize(input = "") {
  let buffer = "";
  const characters = input.split("");
  const tokens = [];

  for (let currentIndex = 0; currentIndex < characters.length; currentIndex++) {
    const currentCharacter = characters[currentIndex];

    // Manipula variáveis proposicionais únicas (P, Q, R, x, y, z)
    if (
      /^[a-zA-Z]$/.test(currentCharacter) &&
      !/^[a-zA-Z]$/.test(characters[currentIndex + 1]) &&
      !/^[a-zA-Z]$/.test(characters[currentIndex - 1])
    ) {
      if (buffer.length > 0) {
        // Processa buffer pendente primeiro
        if (LiteralTokenMap[buffer]) {
          tokens.push(
            createToken(
              LiteralTokenMap[buffer],
              buffer,
              currentIndex - buffer.length
            )
          );
        } else {
          // Verifica se o buffer é prefixo de um token conhecido mas não está completo
          if (isPrefixOfAnyKey(buffer)) {
            throw new LexingError(
              `Sequência incompleta: '${buffer}' (talvez você quis dizer '${Object.keys(
                LiteralTokenMap
              ).find((key) => key.startsWith(buffer))}'?)`,
              currentIndex - buffer.length,
              input
            );
          } else {
            throw new LexingError(
              `Símbolo desconhecido: '${buffer}'`,
              currentIndex - buffer.length,
              input
            );
          }
        }
        buffer = "";
      }

      // Manipula predicados (ex: P(x))
      if (characters[currentIndex + 1] == "(") {
        const predicateStart = currentIndex;
        while (currentIndex < characters.length) {
          buffer += characters[currentIndex];
          if (characters[currentIndex] == ")") break;
          currentIndex++;
        }

        // Validate predicate format
        if (!buffer.endsWith(")")) {
          throw new LexingError(
            `Predicado não fechado: '${buffer}'`,
            predicateStart,
            input
          );
        }

        tokens.push(createToken(TokenType.PREDICATE, buffer, predicateStart));
        buffer = "";
      } else {
        tokens.push(createToken(TokenType.VAR, currentCharacter, currentIndex));
      }

      continue;
    }

    if (currentCharacter === " ") continue; // ignora espaços em branco

    buffer += currentCharacter;

    // Caso de correspondência exata com token conhecido
    if (LiteralTokenMap[buffer]) {
      tokens.push(
        createToken(
          LiteralTokenMap[buffer],
          buffer,
          currentIndex - buffer.length + 1
        )
      );
      buffer = "";
      continue;
    }

    // Se o buffer não é prefixo de nenhum token, trata como desconhecido
    if (!isPrefixOfAnyKey(buffer)) {
      // Verifica se o buffer é um token conhecido que foi particularmente detectado
      const possibleTokens = Object.keys(LiteralTokenMap).filter((key) =>
        key.startsWith(buffer.substring(0, buffer.length - 1))
      );

      if (possibleTokens.length > 0 && buffer.startsWith("\\")) {
        throw new LexingError(
          `Sequência inválida: '${buffer}' (talvez você quis dizer '${possibleTokens[0]}'?)`,
          currentIndex - buffer.length + 1,
          input
        );
      } else {
        throw new LexingError(
          `Símbolo desconhecido: '${buffer}'`,
          currentIndex - buffer.length + 1,
          input
        );
      }
    }
  }

  // Processa qualquer conteúdo restante no buffer ao final
  if (buffer.length > 0) {
    if (LiteralTokenMap[buffer]) {
      tokens.push(
        createToken(
          LiteralTokenMap[buffer],
          buffer,
          input.length - buffer.length
        )
      );
    } else {
      // Verifica se o buffer é prefixo de um token conhecido mas não está completo
      if (isPrefixOfAnyKey(buffer)) {
        throw new LexingError(
          `Sequência incompleta: '${buffer}' (talvez você quis dizer '${Object.keys(
            LiteralTokenMap
          ).find((key) => key.startsWith(buffer))}'?)`,
          input.length - buffer.length,
          input
        );
      } else {
        throw new LexingError(
          `Símbolo desconhecido: '${buffer}'`,
          input.length - buffer.length,
          input
        );
      }
    }
  }

  return new TokenConsumer(tokens, input);
}
