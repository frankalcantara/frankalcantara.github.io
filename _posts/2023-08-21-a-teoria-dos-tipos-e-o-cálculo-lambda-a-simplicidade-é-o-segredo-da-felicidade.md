---
layout: post
title: Teoria dos Tipos, Cálculo Lambda, Simplicidade e Felicidade
author: Frank
categories: []
tags: []
image: ""
featured: 2023-08-21T13:29:48.211Z
rating: 5
description: Introdução ao uso de tipos em Cálculo Lambda
date: 2023-08-21T13:29:48.211Z
preview: ""
keywords: ""
published: false
lastmod: 2025-04-19T00:22:59.127Z
---

Imagine uma biblioteca, vasta e silenciosa. O mais importante repositório do conhecimento humano, mesmo hoje, em tempos de internet, nada se compara ao folhear de um livro. No entanto, só é realmente útil se os livros puderem ser consultados. Nessa nossa biblioteca, de sonhos e lembranças boas, os livros estão organizados em prateleiras e estas em seções. Entre tantas, há uma seção especial. Onde estão os livros puros, humildes, os livros que não se referem a si mesmos.

Nossa bibliotecária, de olhos negros, grandes e profundos, escondidos atrás de óculos de vidro grosso que lhe disfarçam a beleza criando um ar de mistério e erudição, precisa de um catálogo. Justamente da seção dos livros que não citam a si mesmos. Este livro, este catálogo, deve ficar na própria seção. Feita a encomenda do catálogo, o autor do catálogo, arde em dúvidas e pergunta-se repetidamente: o catálogo lista a si mesmo?

O catálogo desta seção deve listar todos os livros que não se referem a si mesmos. Se o catálogo se referir a si mesmo, ele não pertence à seção especial e, portanto, não deve listar-se. Mas se o catálogo não se referir a si mesmo, então ele pertence à seção especial e deve listar-se. Isto é uma contradição.

Pobre da literatura, se perde nos meandros da lógica e da matemática. Talvez possamos entender o problema do escritor do catálogo se abandonarmos a literatura e abraçarmos a matemática. Foi o que Russell fez.

Russell percebeu a contradição analisando um teorema de Cantor que diz que nenhum mapeamento $F:X \rightarrow \text{Pow}(X)$ (onde $\text{Pow}(X)$ é a classe das subclasses de uma classe $X$) pode ser sobrejetivo; isto é, $F$ não pode ser tal que cada membro $b$ de $\text{Pow}(X)$ seja igual a $F(a)$ para algum elemento $a$ de $X$. Isso pode ser expresso _intuitivamente_ como o fato de haver mais subconjuntos de $X$ do que elementos de $X$. Pensa em um parágrafo complicado.

O símbolo $\text{Pow}(X)$ refere-se ao conjunto potência de um conjunto $X$. Esse conjunto potência contém todos os subconjuntos possíveis de $X$, incluindo o próprio conjunto $X$ e o conjunto vazio. Matematicamente, o conjunto potência de um conjunto $X$ é definido como:

$$
\text{Pow}(X) = \{A \mid A \subseteq X\}
$$

Por exemplo, se $X = \{1, 2\}$, então $\text{Pow}(X) = \{\emptyset, \{1\}, \{2\}, \{1, 2\}\}$.

No teorema que Russell estava estudando, $\text{Pow}(X)$ é usado para expressar a ideia de que nenhum mapeamento $F: X \rightarrow \text{Pow}(X)$ pode ser sobrejetivo. Isso é uma parte fundamental da prova que demonstra o paradoxo de Russell e está relacionado à ideia de que há mais subconjuntos de $X$ do que elementos de $X$.

Talvez você consiga fechar o conceito se lembrar que os termos _função sobrejetiva_ e _função sobrejetora_ referem-se à mesma coisa. Uma função é chamada de sobrejetiva, ou sobrejetora, quando cada elemento da imagem, ou codomínio, é a imagem de pelo menos um elemento do domínio. Em outras palavras, a função atinge todos os elementos do codomínio. Isso significa que para uma função $f: A \rightarrow B$, se para todo $b \in B$ existe algum $a \in A$ tal que $f(a) = b$, então a função é sobrejetiva ou sobrejetora.

Vamos considerar um exemplo concreto de uma função sobrejetiva: dado o conjunto de domínio

$A = \{1, 2, 3\}$ e o conjunto codomínio
$B = \{4, 5\}$, vamos definir a seguinte função
$f: A \rightarrow B$:

Teremos:

$f(1) = 4$
$f(2) = 5$
$f(3) = 4$

Neste caso, cada elemento do conjunto codomínio $B$ é mapeado por pelo menos um elemento do conjunto domínio $A$. Especificamente:

O elemento 4 de $B$ é mapeado pelos elementos 1 e 3 de $A$.
O elemento 5 de $B$ é mapeado pelo elemento 2 de $A$.

Portanto, a função $f$ é sobrejetiva, já que todos os elementos do codomínio são alcançados pela função.

Agora, para garantir que veremos tudo em contraste, vamos considerar um exemplo concreto de uma função que não é sobrejetiva. Suponha que temos os conjuntos de domínio

$A = \{1, 2, 3\}$ e codomínio
$B = \{4, 5, 6\}$, e vamos definir a seguinte função
$g: A \rightarrow B$:

$g(1) = 4$
$g(2) = 4$
$g(3) = 5$

Nesse caso, podemos ver que:

O elemento 4 de $B$ é mapeado pelos elementos 1 e 2 de $A$.
O elemento 5 de $B$ é mapeado pelo elemento 3 de $A$.
O elemento 6 de $B$ não é mapeado por nenhum elemento de $A$.

Portanto, a função $g$ não é sobrejetiva, pois existe um elemento em $B$ (neste caso, o elemento representado pelo número 6) que não é mapeado por nenhum elemento de $A$. A falta de um mapeamento para cada elemento de $B$ faz com que a função falhe em ser sobrejetiva.

Voltando ao Teorema de Cantor, podemos tentar prová-lo considere o seguinte subconjunto de $X$:

$$A = \{x \in X \mid x \notin F(x)\}$$

Esse subconjunto não pode estar no alcance de $F$. Pois, se $A = F(a)$, para algum $a$, então
$a \in F(a) \iff a \in A \iff a \notin F(a)$ e, novamente, encontramos a contradição.

[Bertrand Russell](https://en.wikipedia.org/wiki/Bertrand_Russell) discutiu este paradoxo no Appendix B: The Doctrine of Types do seu livro [The Principles of Mathmatics](https://people.umass.edu/klement/pom/) de 1903. Para resolve-lo, Russell sugeriu a criação de um processo de separação de elementos por níveis específicos o que hoje chamamos de Teoria dos Tipos.

Russell separou os conjuntos em níveis, como prateleiras distintas em uma biblioteca. Como os tipos não se misturam, o paradoxo desaparece. O método é chamado de diagonalização. É como ler uma lista na diagonal. Encontra-se coisas que não podem ser listadas.

A diagonalização é um método de prova utilizado para mostrar que certas correspondências ou mapeamentos não podem existir. A ideia básica é argumentar por contradição, criando uma situação em que um suposto mapeamento falha em ter uma propriedade que deveria possuir. Prova por contradição é, geralmente, mais fácil.

A técnica ganha o nome _diagonalização_ por causa de um procedimento comum em que se analisa a diagonal de uma matriz, ou tabela, infinita. O método foi usado por Cantor para provar que não existe uma correspondência um-para-um entre os números naturais e os números reais, tema para outro artigo, e aparece novamente na prova do paradoxo de Russell e no teorema da incompletude de Gödel. Pensa em uma técnica de prova importante!

Funciona, mais ou menos, assim:

1. Suponha que existe um mapeamento de uma certa classe $X$ para outra classe $Y$.
2. Construa um elemento em $Y$ que não pode ser mapeado a partir de $X$, geralmente examinando a _diagonal_ da suposta correspondência.
3. Mostre que essa construção leva a uma contradição, provando que o mapeamento suposto não pode existir.

A beleza da diagonalização é que ela permite trabalhar com conceitos infinitos de forma finita e concreta. A técnica é poderosa e aparece em várias áreas da matemática e da filosofia da matemática.

Na técnica de Russell, a diagonalização não é exatamente igual a de Cantor mas se baseia no mesmo conceito. Como vimos antes:

1. Considere a classe de todos os conjuntos que não pertencem a si mesmos. Podemos representá-la como:
   $$R = \{w \mid w \notin w\}$$

2. Então, pergunte-se: o conjunto $R$ pertence a si mesmo? Isso nos leva a examinar a afirmação:
   $R \in R \iff R \notin R$.

3. A afirmação acima é contraditória: afirma que $R$ pertence a si mesmo se e somente se $R$ não pertence a si mesmo.

Examinamos uma certa _diagonal_. Neste caso, a propriedade de pertencer ou não a si mesmo. E chegamos a uma contradição ao tentar caracterizar a propriedade de um objeto que se auto-referencia. Em resumo, nossa metáfora e nosso paradoxo podem ser resumidos como:

**A Biblioteca e a Classe $X$**: a biblioteca representa o universo de todos os conjuntos, ou, no nosso contexto, a classe $X$.

**Os Livros e $\text{Pow}(X)$**: os livros dentro da biblioteca representam os subconjuntos da classe $X$, que é denotada por $\text{Pow}(X)$.

**O Catálogo e a Função $F$**: o catálogo que tenta listar todos os livros da seção especial é semelhante à função $F:X \rightarrow \text{Pow}(X)$. Ele está tentando mapear cada elemento de $X$ para um subconjunto de $X$, mas enfrenta uma contradição.

**A Seção Especial e o Subconjunto $A$**: a seção especial da biblioteca, que contém livros que não se referem a si mesmos, representa o subconjunto específico $A = \{ x \in X \mid x \notin F(x) \}$. Assim como o catálogo não pode listar a si mesmo sem contradição, esse subconjunto não pode estar no alcance de $F$.

**A Contradição e o Paradoxo de Russell**: a contradição encontrada na tentativa de listar o catálogo em si mesma reflete a contradição matemática expressa na equação $a \in F(a) \iff a \in A \iff a \notin F(a)$. Não podemos consistentemente dizer se $A$ está ou não no alcance de $F$, assim como não podemos consistentemente dizer se o catálogo lista a si mesmo.

**Teoria dos Tipos**: A solução de Russell para essa contradição foi a introdução da teoria dos tipos, que coloca restrições nas formas que os conjuntos podem ser formados e relacionados.

**Método de Diagonalização**: A contradição no exemplo da biblioteca e na expressão matemática acima é um exemplo do método de diagonalização, uma técnica poderosa para provar certas propriedades dos conjuntos, como o fato de que não pode haver uma correspondência um-para-um entre os elementos de um conjunto e seus subconjuntos.

Só não consegui incluir a bibliotecária de olhos negros e profundos. Ninguém é perfeito!

Essa é a essência do Paradoxo de Russell, e é aqui que a teoria dos tipos entra para resolver o problema. A teoria dos tipos impõe uma estrutura hierárquica, onde objetos de um tipo só podem conter ou referir-se a objetos de um tipo inferior. Nessa hierarquia, o catálogo da biblioteca estaria em um tipo superior aos livros que ele lista. Portanto, ele não poderia conter ou referir-se a si mesmo.

À luz da nossa linda metáfora:

- Os livros são objetos de um tipo inferior;
- O catálogo é um objeto de um tipo superior.

Assim, a questão de se o catálogo se cita ou não se torna irrelevante por perder todo sentido. Um catálogo, tipo superior, não pode conter ou referir-se a si mesmo, do mesmo tipo. A hierarquia de tipos impede a formação do paradoxo, restringindo as relações que podem existir entre os objetos.

A contradição é clara. A solução é lógica. A biblioteca é silenciosa novamente, mas a questão permanece. A matemática é simples, a realidade complexa. Uma contradição levou a uma nova compreensão. E a biblioteca permanece lá, intacta, mas mudada. Agora os livros têm tipos diferentes.

1. Paradoxos e Teorias de Tipo de Russell
   A teoria dos tipos foi introduzida por Russell para lidar com algumas contradições que ele encontrou em sua descrição da teoria dos conjuntos e foi introduzida em "Apêndice B: A Doutrina dos Tipos" de Russell 1903.

Alguns comentários são necessários. Primeiro, a prova não usa a lei do terceiro excluído e é assim válida intuicionisticamente. Segundo, o método usado, chamado de diagonalização, já estava presente no trabalho de du Bois-Reymond para construir funções reais que crescem mais rapidamente do que qualquer função em uma sequência de funções.

Russell analisou o que acontece se aplicarmos este teorema ao caso em que A é a classe de todas as classes, admitindo que exista tal classe. Ele foi então levado a considerar a classe especial das classes que não pertencem a si mesmas

(\*)

\( R = \{w \mid w \notin w\} \).

Então temos

\( R \in R \iff R \notin R \).
Parece que Cantor já estava ciente do fato de que a classe de todos os conjuntos não pode ser considerada um conjunto.

Russell comunicou esse problema a Frege, e sua carta, juntamente com a resposta de Frege, aparece em van Heijenoort 1967. É importante perceber que a formulação (\*) não se aplica como está ao sistema de Frege. Como o próprio Frege escreveu em sua resposta a Russell, a expressão "um predicado é predicado de si mesmo" não é exata. Frege tinha uma distinção entre predicados (conceitos) e objetos. Um predicado (de primeira ordem) se aplica a um objeto, mas não pode ter um predicado como argumento. A formulação exata do paradoxo no sistema de Frege usa a noção da extensão de um predicado \(P\), que designamos como \(\varepsilon P\). A extensão de um predicado é ela mesma um objeto. O importante axioma V é:

(Axioma V)

\( \varepsilon P = \varepsilon Q \iff \forall x [P(x) \iff Q(x)] \)
Este axioma afirma que a extensão de \(P\) é idêntica à extensão de \(Q\) se e somente se \(P\) e \(Q\) são materialmente equivalentes. Podemos então traduzir o paradoxo de Russell (\*) no sistema de Frege, definindo o predicado

\( R(x) \iff \exists P [x = \varepsilon P \land \neg P(x)] \)
Pode então ser verificado, usando o Axioma V de forma, que

\( R(\varepsilon R) \iff \neg R(\varepsilon R) \)
e temos uma contradição também. (Observe que, para definir o predicado \(R\), usamos uma quantificação existencial impredicativa sobre predicados. Pode ser mostrado que a versão predicativa do sistema de Frege é consistente (veja Heck 1996 e para mais refinamentos Ferreira 2002).

Está claro a partir desta conta que uma ideia de tipos já estava presente no trabalho de Frege: lá encontramos uma distinção entre objetos, predicados (ou conceitos), predicados de predicados, etc. (Este ponto é enfatizado em Quine 1940.) Essa hierarquia é chamada de "hierarquia extensional" por Russell (1959), e sua necessidade foi reconhecida por Russell como uma consequência de seu paradoxo.
