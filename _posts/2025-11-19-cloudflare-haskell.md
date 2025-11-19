---
layout: post
title: A Falha do Cloudflare e o Haskell
author: Frank
categories:
    - artigo
    - opinião
tags: |
    - haskell
    - cloudflare
    - teoria-de-tipos
    - rust
    - engenharia-de-software
    - programação-funcional
    - post-mortem
    - análise-técnica
    - sql
    - sistemas-distribuídos
rating: 6
description: análise técnica do incidente do Cloudflare em novembro de 2025, explorando como a Teoria de Tipos em Haskell poderia ter prevenido a falha.
date: 2025-11-14T00:58:10.955Z
preview: |
    O incidente ocorrido entre os dias 18 e 19 de novembro representa uma das piores falhas na infraestrutura do Cloudflare desde 2019. O quê e porquê? Leia a análise técnica completa.
lastmod: 2025-11-19T16:08:17.134Z
published: true
draft: 2025-11-19T16:07:35.437Z
image: assets/images/cloudflare2.webp
---

O incidente ocorrido entre os dias 18 e 19 de novembro representa uma das piores falhas na infraestrutura do Cloudflare desde 2019. O evento, que durou alguma coisa entre 3 e 6 horas, ilustra como uma _alteração inocente em permissões de banco de dados pode desencadear uma reação em cadeia catastrófica quando acoplada a otimizações de memória estática em sistemas de alta performance_.

Se a curiosa leitora estiver interessada em saber exatamente como a Cloudflare explicou a falha, recomendo a leitura do [post-mortem oficial](https://blog.cloudflare.com/cloudflare-outage-november-18-2025/). Aqui, porém, pretendo dissecar o incidente sob uma perspectiva do professor, explorando os fundamentos matemáticos e de engenharia que levaram ao colapso, e como a Teoria de Tipos, especialmente em Haskell, poderia ter prevenido tal desastre. Na esperança que pelo menos um dos alunos de linguagem funcional consiga entender o raciocínio por trás do ocorrido. E tenha um momento aha! Finalmente, caiu a ficha, foi isso que ele quis dizer com tipos dependentes!

Para tanto, sem o menor pudor, a seguir, detalho a anatomia da falha, desde a álgebra relacional até a gestão de memória em Rust e o comportamento de sistemas distribuídos, finalizando com uma análise de como a Teoria de Tipos poderia prevenir tais erros.

Teoria dos tipos em Haskell, em Haskell!

### O Vetor da Falha: Álgebra Relacional e Expansão do Conjunto

A raiz do problema foi uma alteração nas ACLs (Listas de Controle de Acesso) de alguns usuários. Um comando SQL, que operava de forma estável há tempos, tinha como objetivo consultar metadados de colunas exclusivamente no banco de dados `default`. E pode ser visto no printscreen abaixo, retirado do post-mortem oficial.

![SQL que criou todo o problema no Cloudflare: SELECT
  name,
  type
FROM system.columns
WHERE
  table = 'http_requests_features'
order by name;](/assets/images/cloudflare1.webp)

Na imagem, o comando SQL consulta a tabela `system.columns`, que armazena metadados sobre as colunas de todas as tabelas em todos os bancos de dados gerenciados pelo sistema. A cláusula `WHERE table = 'http_requests_features'` restringe a consulta para retornar apenas as colunas da tabela `http_requests_features` no banco de dados padrão `default`.

Aqui entra a álgebra, segure-se. 

Se definirmos o conjunto de metadados retornados como $S$, a consulta original visava apenas o esquema $D$:

$$
S_{normal} = \{ x \mid x \in \text{Columns}(D) \}
$$

Neste cenário, a cardinalidade de $|S_{normal}|$ era de aproximadamente 60 itens (features). Valor, também recuperado do post-mortem oficial.

No entanto, a elevação de privilégios expandiu o escopo de visibilidade da consulta para incluir também o banco de dados subjacente $R_0$, uma camada física ou *shard* do `default`(pura simplificação didática). Como $R_0$ é isomórfico a $D$, a consulta passou a retornar a união dos conjuntos sem a devida filtragem de unicidade:

$$
S_{erro} = S_{normal} \cup \{ y \mid y \in \text{Columns}(R_0) \}
$$

Isso resultou em uma duplicação cartesiana efetiva, na qual $|S_{erro}| \approx 120$. O sistema, que não foi projetado prevendo essa alteração de escopo, não possuía cláusulas de `DISTINCT` ou filtros de *schema* para tratar essa redundância.

Se a amável leitora se perdeu, pode ser necessário revisar conceitos básicos de álgebra relacional, especialmente operações de união e projeção. Talvez seja a hora de deixar um pouco de lado o SQL e tentar entender o que esta linguagem realmente representa em termos matemáticos.

>Problemas como esse poderiam ser evitados se estudássemos um pouco mais de matemática e um pouco menos de linguagens de programaçao. Não, eu não consigo segurar o professor dentro de mim.

### O Colapso: Rust, `Panic` e Limites Hardcoded

O Cloudflare utiliza um módulo de *anti-crawling* (anti-rastreamento) otimizado para altíssima performance. Para evitar a latência de alocação dinâmica de memória (o custo de *syscalls* como `malloc` e `free`) durante o processamento de requisições, os engenheiros optaram por alocação estática ou pré-alocada.

Havia um limite *hardcoded* (fixo no código) de **200 features**. Novamente, do post-mortem oficial.

Embora o conjunto de dados duplicado ($\approx 120$) pareça numericamente inferior ao limite de 200, a estrutura de dados resultante, o arquivo de recursos gerado, violou as restrições de integridade ou tamanho do buffer, levando ao estouro do limite lógico implementado.

Em linguagens como C++, um estouro de buffer poderia resultar em corrupção de memória silenciosa ou execução de código arbitrário. E aqui mora 100% do perigo que o Cloudflare tentou evitar usando o Rust. Nunca, jamais, em tempo algum, permitir que dados corrompidos ou inesperados possam levar a falhas de segurança. Por isso o Rust.

O Rust, focado em *memory safety*, adota uma postura diferente. Ao detectar a violação do limite (*bounds check*), o *runtime* executa um `panic!`:

```rust
// Representação conceitual do mecanismo de falha
if features.len() > MAX_FEATURES { // MAX_FEATURES = 200
    panic!("Feature limit exceeded"); // Aborta a thread/processo imediatamente
}
```

Isso não é um erro. A linguagem, ainda que muito nova, foi projetada para falhar rápido e alto em situações de violação de segurança. 

Esse mecanismo de defesa, ao ser acionado sistemicamente, fez com que o serviço retornasse erros 500 (Internal Server Error) em escala global.

### Propagação e Oscilação (Flapping)

O cenário tornou-se caótico devido à arquitetura distribuída de atualização. O arquivo de recursos é gerado automaticamente a cada 5 minutos e propagado para a *edge* (borda). A camada da infraestrutura de computação que está fisicamente e logicamente mais próxima do usuário final,

A atualização das permissões no banco de dados não foi atômica em todos os nós; ela ocorreu via *rollout* gradual. Uma distribuição progressiva por todo o sistema. Isso criou um estado de inconsistência eventual no cluster de banco de dados. A informação do post-mortem oficial sugere que a propagação levou várias horas para ser concluída.

Novamente podemos ver essa situação com a boa e velha matemática de conjuntos. Seja $N$ o conjunto de nós de banco de dados:

* **Nós $N_{antigos}$:** Ainda com permissões antigas $\rightarrow$ geram arquivos válidos.
* **Nós $N_{novos}$:** Com novas permissões $\rightarrow$ geram arquivos corrompidos/excessivos.

Como o sistema de geração do arquivo consultava os nós de forma balanceada, provavelmente usando um algoritmo parecido com o *Round-Robin*, a saúde do sistema global $H(t)$ tornou-se uma variável estocástica dependente de qual nó servia a consulta naquele instante $t$:

$$
P(\text{Falha}) = \frac{|N_{novos}|}{|N_{total}|}
$$

Isso gerou o fenômeno de **oscilação** (*flapping*): o sistema da Cloudflare ficou oscilando entre operação normal e colapso total a cada ciclo de geração, levando os engenheiros a suspeitarem incorretamente de um ataque DDoS. E nessa suspeita está a origem do boato que circulou nas redes sociais. Levando vários órgãos de imprensa a embarcar no bote furado e afundar junto com o boato. Novamente, outra vez, de novo. A imprensa não aprende.

Este não foi o único problema. Coincidências não existem. Aprendi isso com meu amigo Gibbs. Ou seja, acredite se quiser.

### A Coincidência e o Diagnóstico

Para dificultar a análise, a página de status do Cloudflare, hospedada em infraestrutura de terceiros longe da engenharia da empresa, falhou simultaneamente, reforçando a hipótese de um ataque externo coordenado.

>Nesse ponto da história, eu devo admitir, que também teria pulado no mesmo barco e afundado junto com eles. A não ser que eu lembrasse do Gibbs. Acho que seria interessante calcular as probabilidades de duas falhas independentes acontecerem ao mesmo tempo.

>Porém, sempre tem um porém. Uma única infraestrutura para monitor o status da maior empresa do mundo de **DISTRIBUIÇÃO DE CONTEÚDO**? Sério isso? Isso é tão 1990s.

Essa correlação espúria desviou o foco da equipe de engenharia da causa raiz, lógica interna para mitigação de ataques. E fez com que eles corressem atrás do próprio rabo por horas.

A estabilidade do erro só foi alcançada quando a propagação das permissões foi concluída para $100\%$ dos nós ($|N_{novos}| = |N_{total}|$), momento em que o sistema falhou permanentemente, permitindo o isolamento da variável causal e a subsequente correção. Ficou bonito isso: variável causal. Gostei.

## Repita Comigo: "Parse, don't Validate"

Na engenharia de software tradicional, frequentemente adotamos o padrão de **validação**: lemos um dado, verificamos se ele obedece a uma regra (ex: `len < 200`) e, se não obedecer, lançamos uma exceção ou causamos um *panic*. O problema dessa abordagem é que a validação é apenas uma verificação em tempo de execução, dissociada do sistema de tipos. O compilador não sabe que aquele dado foi validado.

Em Haskell, e na teoria de linguagens de programação moderna, aplicamos o mantra: **Parse, não valide**.

>Não gosto de dizer isso, mas eu te disse!

A ideia é que não devemos apenas verificar uma condição e retornar o mesmo tipo de dado. Devemos transformar o dado bruto em um tipo distinto que **garante estruturalmente** a invariante desejada. Se o dado é inválido, é impossível construir o tipo, forçando o tratamento do erro na fronteira do sistema.

### Modelagem do Problema em Haskell

No cenário do Cloudflare, o problema foi um `Vector` contendo mais elementos do que a memória pré-alocada suportava. Em Rust, a verificação manual falhou e resultou em `panic`.

Em Haskell, usaríamos um **Smart Constructor** para criar um tipo opaco. Vamos definir que nosso conjunto de *features* não é apenas uma lista de strings, mas sim um tipo `BoundedFeatureSet`.

```haskell
module Cloudflare.AntiBot 
    ( FeatureSet         -- Exporta o Tipo
    , mkFeatureSet       -- Exporta apenas o construtor seguro
    , getFeatures        -- Acessor
    ) where

import qualified Data.Vector as V
import Data.Text (Text)

-- Definição de constantes de engenharia
maxFeatures :: Int
maxFeatures = 200

-- O Tipo é "Opaco": o usuário externo não pode usar o construtor 'FeatureSet'
-- diretamente, impedindo a criação de estados inválidos manualmente.
newtype FeatureSet = FeatureSet (V.Vector Text)
    deriving (Show, Eq)

-- Smart Constructor: A única porta de entrada
-- Retorna um Either: Força quem chama a lidar com o erro EXPLICITAMENTE.
mkFeatureSet :: V.Vector Text -> Either String FeatureSet
mkFeatureSet rawData
    | V.length rawData > maxFeatures = Left "Erro Crítico: Limite de features excedido (Potencial corrupção de Schema)"
    | otherwise                      = Right (FeatureSet rawData)

getFeatures :: FeatureSet -> V.Vector Text
getFeatures (FeatureSet fs) = fs
```

### O Sucesso está nos Pequenos Detalhes

A mudança sutil, mas poderosa, acontece no consumo dessa API.

No código original (Rust com `panic` ou C++ sem tratamento), o fluxo é imperativo:

$$
f: \text{Dados} \rightarrow \text{Void} \quad (\text{com efeito colateral maligno})
$$

No modelo funcional, Haskell:

$$
f: \text{Dados} \rightarrow \text{Either Error FeatureSet}
$$

O compilador obriga o desenvolvedor a "desembrulhar" o resultado. Não é possível acessar os dados do `FeatureSet` sem antes verificar se ele é um `Right` (sucesso) ou `Left` (falha).

> Não faz nem uma semana que meus alunos entregaram um trabalho usando exatamente esse padrão. Atenção: momento aha! se aproximando.

### Exemplo de Consumo Seguro

Imagine a função que gera o arquivo de configuração para os servidores globais. Com o modelo acima, o código seria algo assim:

```haskell
-- Simulação da função principal que processa a query do banco
processDatabaseResult :: V.Vector Text -> IO ()
processDatabaseResult rawRows = do
    -- Tenta elevar os dados brutos para o tipo seguro
    let result = mkFeatureSet rawRows
    
    case result of
        -- Caso de Sucesso: O sistema continua garantidamente seguro
        Right safeFeatures -> do
            pushToGlobalEdge (optmizeMemory safeFeatures)
            logInfo "Configuração atualizada com sucesso."

        -- Caso de Falha: O BUG DO SQL É CAPTURADO AQUI
        -- Em vez de derrubar o servidor (Panic/500), entramos num estado de fallback.
        Left err -> do
            logError $ "Falha de integridade detectada: " ++ err
            keepPreviousConfiguration -- Mantém a última config válida (Graceful Degradation)
            alertEngineeringTeam      -- Aciona o time sem derrubar o serviço
```

### 6. Análise Formal do Impacto

Se analisarmos sob a ótica da Teoria das Categorias, transformamos uma função parcial, que não é definida para todo $x$, pois falha para $|x| > 200$, em uma função total, definida para todo $x$, mapeando para um codomínio `Either`.

Seja $D$ o conjunto de todos os vetores possíveis de strings retornados pelo SQL.
Seja $V \subset D$ o subconjunto de vetores válidos onde o comprimento $\le 200$.

A abordagem da equipe da Cloudflare assumiu implicitamente que o SQL sempre retornaria $x \in V$. Quando o SQL retornou $y \notin V$, a função de projeção para a memória falhou.

A abordagem com Tipos Dependentes, ou *Smart Constructors em Haskell*, define a função de processamento $P$ apenas sobre o tipo refinado $T_V$, onde $T_V$ é isomórfico a $V$.

$$
P: T_V \rightarrow \text{SystemState}
$$

Como o dado vindo do banco ($y$) não pôde ser convertido para $T_V$, o construtor retornou `Left`, a função $P$ nunca foi invocada com dados corrompidos. O sistema de tipos serviu como um "firewall lógico", impedindo que o erro de dados (SQL) se transmutasse em um erro de infraestrutura (Crash).

## Apontando o Dedo 

A culpa não foi do pobre desenvolvedor que alterou as regras de busca do malfadado SQL. OK, foi sim. Mas a culpa maior é do modelo mental que adotamos na engenharia de software moderna, que se dedica a linguagem de programação e esquece a matemática por trás dela.

Se os desenvolvedores soubessem modelar o problema e conhecessem a matemática que fundamenta as linguagens de programação, desastres como esse nunca aconteceriam.

Se, em vez de ensinarmos como fazer um `join` ensinássemos porque fazer um `join`, talvez os engenheiros pudessem antecipar os efeitos colaterais de uma alteração aparentemente inocente.

A Teoria de Tipos não é apenas um conceito acadêmico distante. Ela é uma ferramenta prática e poderosa para construir sistemas robustos, seguros e resilientes.

Talvez, só talvez, se mais engenheiros de software, desenvolvedores, programadores, professores e estudantes entendessem a matemática leríamos e ouviríamos menos besteiras como: linguagens de programação seguras.

Finalmente, fica um convite. Semestre sim, semestre não, eu ministro uma disciplina de programação funcional, na Pontifícia Universidade Católica do Paraná. Se você é estudante de engenharia de software, ciência da computação ou áreas afins, considere se inscrever. Garanto que será um divisor de águas na sua carreira.

Além disso estou no X (antigo Twitter) como [@frankalcantara](https://x.com/frankalcantara). Sinta-se à vontade para me seguir e discutir mais sobre esses temas fascinantes.