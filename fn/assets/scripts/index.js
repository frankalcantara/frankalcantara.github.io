import { tokenize } from "./modules/lexer.js";
import { parse } from "./modules/parser.js";
import { executeCompleteTransformations } from "./modules/transformer.js";

// Definições de macros customizadas para KaTeX
const customMacros = {
  "\\neg": "\\mathord{\\sim}",
  "\\land": "\\wedge",
  "\\lor": "\\vee",
  "\\rightarrow": "\\to",
  "\\leftrightarrow": "\\Leftrightarrow",
  "\\lnot": "\\mathord{\\sim}",
  "\\iff": "\\Leftrightarrow",
  "\\implies": "\\to",
};

// Passos destacados na interface
const HIGHLIGHTED_STEPS = [
  "Forma Normal Negativa",
  "Forma Normal Conjuntiva Prenex",
  "Forma Normal Disjuntiva Prenex",
  "Forma Normal Skolem",
  "Forma Normal Conjuntiva",
  "Forma Normal Disjuntiva",
  "Forma Clausal",
  "Cláusula de Horn",
];

// Gerenciamento de tema (claro/escuro)
function initializeTheme() {
  const savedTheme = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme");
  const newTheme = currentTheme === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", newTheme);
  localStorage.setItem("theme", newTheme);
  updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
  const themeIcon = document.getElementById("theme-icon");
  if (theme === "dark") {
    themeIcon.className = "fas fa-sun";
  } else {
    themeIcon.className = "fas fa-moon";
  }
}

// ================= Função para Renderizar LaTeX =================
function renderLatex(latexText, targetElement) {
  try {
    targetElement.innerHTML = ""; // limpa conteúdo anterior
    if (!latexText.trim()) {
      targetElement.innerHTML = `<span style="color: var(--text-light);">Sua fórmula aparecerá aqui</span>`;
      targetElement.classList.remove("has-content");
      return;
    }

    const containerElement = document.createElement("div");
    containerElement.className = "katex-container";

    // Usa KaTeX para renderizar o LaTeX
    katex.render(latexText, containerElement, {
      displayMode: true,
      throwOnError: false,
      macros: customMacros,
    });

    targetElement.appendChild(containerElement);
    targetElement.classList.add("has-content");
  } catch (error) {
    console.error(error);
    targetElement.innerHTML = `<div class="text-center p-4" style="color: var(--accent);"><i class="fas fa-exclamation-circle mr-2"></i>Erro: ${error.message}</div>`;
    targetElement.classList.remove("has-content");
  }
}

// ================= Exibir Passos de Transformação =================
function displayTransformationSteps(steps) {
  const stepsContainer = document.getElementById("steps-content");

  // Limpa conteúdo anterior
  stepsContainer.innerHTML = "";

  // Cria grid para os passos
  const gridContainer = document.createElement("div");
  gridContainer.className = "steps-grid";
  stepsContainer.appendChild(gridContainer);

  // Cria cards para cada passo
  steps.forEach((step, index) => {
    const isHighlighted = HIGHLIGHTED_STEPS.includes(step.name);

    const stepCard = document.createElement("div");
    stepCard.className = `step-card ${isHighlighted ? "highlighted" : ""}`;

    // Define delay de animação para efeito escalonado
    stepCard.style.animationDelay = `${index * 0.1}s`;

    stepCard.innerHTML = `
      <div class="flex items-center mb-2">
        <span class="step-number">${index + 1}</span>
        <h3 class="text-md font-semibold">${step.name}</h3>
      </div>
      <div class="step-content">
        <button class="copy-btn" data-formula="${step.result.replace(
          /"/g,
          "&quot;"
        )}">
          <i class="fas fa-copy"></i>
        </button>
        <div class="step-result" id="step-${index}"></div>
      </div>
    `;

    gridContainer.appendChild(stepCard);

    // Renderiza o LaTeX para este passo com um pequeno delay
    setTimeout(() => {
      renderLatex(step.result, document.getElementById(`step-${index}`));
    }, 100 + index * 100);
  });

  // Adiciona event listeners de cópia após um delay para garantir que elementos estão renderizados
  setTimeout(() => {
    document.querySelectorAll(".copy-btn").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const formula = button.getAttribute("data-formula");
        copyToClipboard(formula);
      });
    });
  }, 500);
}

// ================= Copiar para Área de Transferência =================
function copyToClipboard(text) {
  navigator.clipboard
    .writeText(text)
    .then(() => {
      // Mostra notificação toast
      const toastElement = document.getElementById("toast");
      toastElement.classList.add("show");
      setTimeout(() => {
        toastElement.classList.remove("show");
      }, 2000);
    })
    .catch((error) => {
      console.error("Falha ao copiar texto: ", error);
    });
}

// ================= Exibir Erro com Indicador de Posição =================
function displayErrorWithIndicator(error, formula) {
  const stepsContainer = document.getElementById("steps-content");

  // Se for um ParsingError, mostra a mensagem com indicador de posição
  if (error.displayMessage) {
    stepsContainer.innerHTML = `
      <div class="error-container">
        <div class="error-header">
          <i class="fas fa-exclamation-circle"></i>
          <h3>Erro de Sintaxe</h3>
        </div>
        <div class="error-message">${error.message}</div>
        <div class="error-location">
          <pre class="formula-display">${formula}</pre>
          <pre class="error-indicator">${error.displayMessage
            .split("\n")
            .slice(-2)
            .join("\n")}</pre>
        </div>
      </div>
    `;
  } else {
    // Para outros tipos de erro, mostra a mensagem simples
    stepsContainer.innerHTML = `
      <div class="text-center p-4" style="color: var(--accent);">
        <i class="fas fa-exclamation-circle mr-2"></i>
        Erro: ${error.message}
      </div>`;
  }
}

// ================= Event Listeners =================
const latexInput = document.getElementById("latex-input");
const latexOutput = document.getElementById("latex-output");
const calculateButton = document.getElementById("calculate-btn");
const exampleChips = document.querySelectorAll(".example-chip");
const themeToggleButton = document.getElementById("theme-toggle");

// Inicializa tema
initializeTheme();

// Toggle de tema
themeToggleButton.addEventListener("click", toggleTheme);

// Preview ao vivo de LaTeX
latexInput.addEventListener("input", () => {
  renderLatex(latexInput.value, latexOutput);
});

// Chips de exemplo
exampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    latexInput.value = chip.dataset.formula;
    renderLatex(latexInput.value, latexOutput);
  });
});

// Botão de calcular
calculateButton.addEventListener("click", () => {
  const formula = latexInput.value.trim();
  if (!formula) return;

  // Mostra estado de carregamento
  const stepsContainer = document.getElementById("steps-content");
  stepsContainer.innerHTML = `
    <div class="text-center p-8" style="color: var(--text-light);">
      <i class="fas fa-spinner fa-spin text-xl mb-3" style="color: var(--primary);"></i>
      <p>Processando transformações<span class="loading-dots"><span></span><span></span><span></span></span></p>
    </div>`;

  try {
    const tokenConsumer = tokenize(formula);
    const abstractSyntaxTree = parse(tokenConsumer);
    const transformationSteps =
      executeCompleteTransformations(abstractSyntaxTree);

    displayTransformationSteps(transformationSteps);
  } catch (error) {
    console.error(error);
    displayErrorWithIndicator(error, formula);
  }
});

// Renderização inicial
document.addEventListener("DOMContentLoaded", () => {
  renderLatex(latexInput.value, latexOutput);
});
