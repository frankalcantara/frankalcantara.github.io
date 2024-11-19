---
layout: post
title: Os desafios da norma IEEE 754 na computação moderna
author: Frank
categories:
  - artigo
  - Matemática
  - computação
tags:
  - interpretadores
  - estrutura de dados
  - modelagem
image: assets/images/pontosflutu.webp
preview: um estudo novo sobre uma das normas mais importantes e menos conhecidas de toda a ciência da computação.
featured: false
rating: 3.5
slug: precisao-realidade-os-desafios-da-norma-ieee-754-na-computacao-moderna
lastmod: 2024-11-19T02:07:38.279Z
---

A memória é escassa, limitada, insuficiente e inteira. O arredondamento de números reais é inevitável, levantando um dilema sobre a extensão da informação a ser armazenada e os métodos de armazenamento. A eficiência computacional é primordial na solução dos problemas concretos que enfrentamos todos os dias. A utilização de normas para a representação números reais na forma de ponto flutuante surgiu como uma resposta. Este artigo desvelará sua funcionalidade e os desafios que esta representação impõe.

<span>Foto de <a href="https://unsplash.com/@calliestorystreet?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Callie Morgan</a> on <a href="https://unsplash.com/s/photos/floating?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

Este problema de armazenamento não é exclusivo dos computadores, o caderno que a leitora usou para aprender a somar era limitado em quantidade de linhas por página e quantidade de páginas por tomo. Nem tudo poderia ser escrito e para tudo havia um custo. Na quantidade de linhas, páginas, tempo de escrita, tempo de localização e tempo de recuperação. Um poema, uma equação, uma resposta, escritos para serem úteis teriam que ser recuperados, lidos e entendidos. No oceano de possibilidades que constitui a computação, não é diferente.

Para que fique claro vamos começar nos concentrando nos números decimais, os números escritos na base $10$. Entre todos os números possíveis na base $10$, estamos particularmente interessados no conjunto dos Números $reais, $\mathbf{R}$. Os números reais englobam um conjunto vasto de números que incluem:

1. **Números Inteiros**: são números que não têm partes fracionárias, como -3, 0, 1, 2, etc.
2. **Números Racionais**: são números que podem ser expressos como uma fração de inteiros, por exemplo, $\frac{3}{4}$, $\frac{5}{2}$, etc.
3. **Números Irracionais**: são números que não podem ser expressos como uma fração de inteiros e têm uma sequência infinita e não periódica de dígitos após a vírgula decimal. Exemplos são $\pi$ e $\sqrt{2}$.
4. **Números Decimais**: são números que têm uma parte decimal finita ou uma sequência infinita periódica de dígitos após a vírgula decimal.

Os **números reais fracionários**, por outro lado, são um subconjunto dos números reais que podem ser expressos como uma fração de dois inteiros, ou seja, um número racional. Eles podem ser representados na forma de uma fração $\frac{a}{b}$, onde "a" é o numerador e "b" é o denominador (e "b" não é igual a zero). Eles também podem ser representados como decimais finitos ou decimais periódicos. Por exemplo:

$$0,125 = \frac{1}{10}+\frac{2}{100}+\frac{5}{1000} = \frac{1}{10^1}+\frac{2}{10^2}+\frac{5}{10^3}$$

> "Deus criou os inteiros, todo o resto é trabalho dos homens." Leopold Kronecker

Não concordo muito com [Kronecker](https://en.wikipedia.org/wiki/Leopold_Kronecker). Acho que Deus criou os números naturais, até os números inteiros devem ser cobrados da humanidade. Todos os números fora do conjunto dos Números Naturais, $\mathbb{N}$, estão envolvidos em uma névoa indefinida de teoremas, axiomas e provas matemáticas para explicar sua existência. Nós os criamos, e não podemos mais viver sem eles.

Infelizmente, errar é humano e, além disso, a exatidão na representação de números reais através de operações fracionárias é uma ocorrência rara. Isto significa que a representação de números reais, não pode ser completamente realizada, usando números inteiros, mesmo que recorramos ao uso de frações para representar a parte fracionária, ou decimal. Esta incompletude na representação de números reais terá um impacto imprevisto e abrangente em todos os sistemas computacionais, desenvolvidos com base nas ideias de [Touring](https://en.wikipedia.org/wiki/Alan_Turing). Lembre-se, em uma célula da memória de um computador existe um número binário, um número do Conjunto dos Números Inteiros, $\mathbb{Z}$, escrito na base $2$.

Vamos ficar um pouco mais na base decimal, para tentar explicar melhor este problema. Tome, por exemplo, a razão $\frac{1}{6}$ e tente representá-la em números reais sem arredondar, ou truncar. Esqueça a calculadora e o computador por um momento.Pegue um lápis e uma folha de papel e tente. Tem pressa não! Eu espero.

Se a leitora tiver tentado, terá visto, muito rapidamente, que seremos forçados a parar a divisão e arredondar, ou truncar o resultado. Obtendo, invariavelmente, algo como $0,166667$. O ponto em que paramos determina a precisão que usaremos para representar este número e a precisão será, por sua vez, imposta, ou sugerida, apenas pelo uso que daremos a este número. Nesta sentença a palavra _uso_ é a mais importante. É Este _uso_ que definirá o modelo que usaremos para resolver um problema específico. Todos os problemas são diferentes, todos os modelos serão diferentes.

Voltando ao nosso exemplo: fizemos a divisão representada por $\frac{1}{6}$ e encontramos $0,166667$. A multiplicação é a operação inversa de divisão. Logo se multiplicarmos $0,166667 \times 6$ deveríamos encontrar $1$ contudo encontramos: $1.000002$. Um erro de $0.000002$. No seu caderno, prova, ou cabeça, isso é $1$, mas só nestes lugares específicos e fora do alcance dos outros seres humanos. Triste será a sina daquele que não perceber que $1.000002$ é muito diferente de $1$.

Em uma estrada, a diferença de um centímetro que existe entre $12,00 m$ e $12,01 m$ provavelmente não fará qualquer diferença no posicionamento de um veículo. Se estivermos construindo um motor à gasolina, por outro lado, um erro de $1 cm$ será a diferença entre o funcionamento e a explosão. Maximize este conceito imaginando-se no lugar de um um físico que precise utilizar a constante gravitacional. Neste caso, a leitora enfrentará a aventura de fazer contas com números como tão pequenos quanto $0.00000000006667_{10}$.

Graças ao hardware que criamos nos últimos 100 anos, números reais não são adequados ao uso em computação. Pronto falei!

Nossos computadores são binários, trabalham só, e somente só, com números na inteiros na base $2$. Sem pensar muito dá para perceber que existe um número infinito de números reais, representados por um número também infinito de precisões diferentes e que, para que os computadores sejam uteis, todo este universo teve que ser colocado em um espaço restrito definido pela memória disponível e pelas regras da aritmética inteira binária. Não precisa ficar assustada, mas se estiver pensando em ficar assustada a hora é essa.

Assim como os números na base dez, os números reais na base dois podem ser representados por uma parte inteira e uma parte fracionária. Vamos usar o número $0.001_{2}$ como exemplo. Este número pode ser representado por uma operação de frações. Para isso, basta considerar a base $2$:

$$0,001 = \frac{0}{2}+\frac{0}{4}+\frac{1}{8} = \frac{0}{2^1}+\frac{0}{2^2}+\frac{1}{2^3}$$

Novamente, sou portador de notícias ruins. Os números fracionários na base $2$ padecem da mesma dor que os números reais na base $10$. A maioria dos números binários facionários, não pode ser representada de forma exata por uma operação de frações. Não bastando isso, a conversão entre as bases $10$ e $2$, acaba criando números binários que não têm fim. Um bom exemplo pode ser visto com a fração $\frac{1}{3}$ que seria representada, em conversão direta para o binário, por $(\frac{1}{11})_2 = 0.0101010101010101_2$ este valor terá que ser arredondado, ou truncado. Esta conversão pode ser vista na Tabela 1:

<table class="table table-striped">
  <tr>
    <th>Passo</th>
    <th>Operação</th>
    <th>Resultado Decimal</th>
    <th>Parte Inteira</th>
    <th>Parte Fracionária (Binário)</th>
  </tr>
  <tr>
    <td>1</td>
    <td>$1 \div 3$</td>
    <td>0.3333...</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>2</td>
    <td>$0.3333... \times 2$</td>
    <td>0.6666...</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>3</td>
    <td>$0.6666... \times 2$</td>
    <td>1.3333...</td>
    <td>1</td>
    <td>01</td>
  </tr>
  <tr>
    <td>4</td>
    <td>$0.3333... \times 2$</td>
    <td>0.6666...</td>
    <td>0</td>
    <td>010</td>
  </tr>
  <tr>
    <td>5</td>
    <td>$0.6666... \times 2$</td>
    <td>1.3333...</td>
    <td>1</td>
    <td>0101</td>
  </tr>
  <tr>
    <td>6</td>
    <td>$0.3333... \times 2$</td>
    <td>0.6666...</td>
    <td>0</td>
    <td>01010</td>
  </tr>
  <tr>
    <td>7</td>
    <td>$0.6666... \times 2$</td>
    <td>1.3333...</td>
    <td>1</td>
    <td>010101</td>
  </tr>
  <tr>
    <td>8</td>
    <td>$0.3333... \times 2$</td>
    <td>0.6666...</td>
    <td>0</td>
    <td>0101010</td>
  </tr>
  <tr>
    <td>9</td>
    <td>$0.6666... \times 2$</td>
    <td>1.3333...</td>
    <td>1</td>
    <td>01010101</td>
  </tr>
  <tr>
    <td>10</td>
    <td>$0.3333... \times 2$</td>
    <td>0.6666...</td>
    <td>0</td>
    <td>010101010</td>
  </tr>
</table>
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Tabela 1 - Conversão de $(\frac{1}{3})_{10}$ em binário.</legend>

Definir o ponto onde iremos parar a divisão, determinará a precisão com que conseguiremos representar o valor $(\frac{1}{11})_2$. Além disso, precisaremos encontrar uma forma de armazenar esta representação em memória.

No exemplo dos valores na base decimal que vimos ante, a leitora aprendeu que os valores que aparecem depois da vírgula e que se repetem até o infinito são chamados de dízima, ou dízima periódica. Se por "dízima" entendemos uma sequência _que não terminará_, então tais números decimais não existem em binário, para que eles existam teremos que parar a divisão e criar uma versão deste número com precisão limitada.

Todos os números reais na base dez, que sejam dízimas, quando representados em binário, também terão repetições infinitas de dígitos. Contudo, há um agravante, muitos números reais exatos, quando convertidos em binário resultam e números com repetições infinitas depois da vírgula.

Só para lembrar: a memória é limitada e contém números inteiros, nosso problema é encontrar uma forma de representar todo o universo de números reais, em base $10$, em um espaço limitado de memória em base $2$. Se pensarmos em uma correspondência de um para um, todo e qualquer número real deve ser armazenado no espaço de dados definido por um e apenas um endereço de memória. Aqui a leitora há de me permitir adiantar um pouco as coisas: esta representação é impossível.

<img class="img-fluid" src="{{ site.baseurl }}/assets/images/churchill1.webp" alt="mostra as distribuição de bits o padrão ieee 754">

## Lá vem o homem com suas imperfeições

Em 1985 o _Institute of Electrical and Electronics Engineers_ (IEEE) publicou uma norma, a norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) cujo objetivo era padronizar uma representação para números de ponto flutuante que deveria ser adotada pelos fabricantes de software e hardware. Na época, os dois mais importantes fabricantes de hardware, Intel e Motorola, apoiaram e adotaram esta norma nas suas máquinas isso foi decisivo para a adoção que disseminada temos hoje. Para os nós interessa que a norma IEEE 754 descreve com representar números com binários com precisão simples, $32 bits$, dupla, $64 bits$, quádrupla $128 bits$ e óctupla $256 bits$. Esta representação é complexa, fria e direta. Talvez fique mais fácil se começarmos lembrando o que é uma notação científica.

Na matemática e nas ciências, frequentemente nos deparamos com números muito grandes ou muito pequenos. Para facilitar a representação e manipulação desses números, utilizamos a **notação científica**, uma forma especial de expressar números em base $10$. Nesta notação, um número é representado por duas partes: a mantissa e o expoente:

- A mantissa é a parte significativa do número, que contém os dígitos mais importantes do número que estamos representando.

- O expoente, $e$, indica a potência a qual a base $10$ deve ser elevada para obter o número original. Assim, a representação geral de um número em notação científica é dada por $\text{mantissa} \times 10^e $. 
  
Para exemplos desta representação veja a Tabela 1.

<table class="table table-striped" style="text-align: center;">
  <thead>
    <tr>
      <th>Mantissa</th>
      <th>Expoente</th>
      <th>Notação Científica</th>
      <th>Valor em Ponto Fixo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$2,7$</td>
      <td>$4$</td>
      <td>$2,7 \times 10^4$</td>
      <td>$27.000$</td>
    </tr>
    <tr>
      <td>$-3.501$</td>
      <td>$2$</td>
      <td>$-3.501 \times 10^2$</td>
      <td>$-350.1$</td>
    </tr>
    <tr>
      <td>$7$</td>
      <td>$-3$</td>
      <td>$7 \times 10^{-3}$</td>
      <td>$0.007$</td>
    </tr>
    <tr>
      <td>$6,667$</td>
      <td>$-11$</td>
      <td>$6,667\times 10^{-11}$</td>
      <td>$0,00000000006667$</td>
    </tr>
  </tbody>
</table>
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Tabela 2 - Exemplos de representação de valor em notação científica.</legend>

Uma boa prática no uso da notação científica é deixar apenas um algarismo antes da vírgula e tantos algarismos significativos quanto necessário para o cálculo específico que pretendemos realizar depois da vírgula. Escolhemos a quantidade de números significativos de acordo com a aplicação. Estes algoritmos depois da vírgula terão impacto na precisão do seu cálculo. O $\pi$, com sua infinitude de dígitos depois da vírgula, é um bom exemplo de precisão relativa à aplicação.

Normalmente, um engenheiro civil, ou eletricista, usa o $\pi$ como $3.1416$. Assim mesmo! Arredondando na última casa, pecado dos pecados. A verdade é que quatro algarismos significativos depois da vírgula são suficientemente precisos para resolver a maior parte dos problemas que encontramos no mundo sólido, real, visível e palpável.

Em problemas relacionados com o eletromagnetismo normalmente usamos $\pi = 3.1415926$, igualmente arredondando a última casa mas com $7$ algarismos significativos depois da vírgula. Em problemas relacionados com o estudo da cosmologia usamos $\pi = 3.14159265359$, truncado, sem nenhum arredondamento, com onze algarismos significativos depois da vírgula. Em física de partículas, não é raro trabalhar com 30 dígitos de significativos para $\pi$. A leitora, amável e paciente, pode ler um pouco mais sobre a quantidade de dígitos significativos necessários [lendo um artigo do Jet Propulsion Lab](https://www.jpl.nasa.gov/edu/news/2016/3/16/how-many-decimals-of-pi-do-we-really-need/). 

O melhor uso da notação científica determina o uso de um, e somente um, algarismo antes da vírgula. Além disso, _a norma impõe que você não deve usar o zero como único algarismo antes da vírgula_. Adotando estas duas regras, $3.1416$ poderia ser representado por $3.1416 \times 10^0$, o que estaria perfeitamente normalizado, ou por $31,416\times 10^{-1}$, uma representação matematicamente válida, mas não normalizada. É importante não esquecer que **números que têm $0$ como sua parte inteira não estão normalizados**.

Passou pela minha cabeça agora: está claro que a nomenclatura _ponto flutuante_ é importada do inglês? Se fosse em bom português, seria _vírgula flutuante_. Esta é uma daquelas besteiras que fazemos. Vamos falando, ou escrevendo, estas coisas, sem nos darmos conta que não faz sentido no idioma de [Mário Quintana](https://en.wikipedia.org/wiki/M%C3%A1rio_Quintana). Herança colonial. Quem sabe?

A base numérica, decimal, hexadecimal, binária, não faz nenhuma diferença na norma da notação científica. Números binários podem ser representados nesta notação tão bem quanto números decimais ou números em qualquer outra base. A leitora pode, por exemplo, usar o número $43.625_{10}$ que, convertido para binário, seria $101011,101_2$ e representá-lo em notação científica como $1,01011101 \times 2^5$. Guarde este número, vamos precisar dele em uma discussão posterior. Sério, guarde mesmo.

> "Idealmente, um computador deve ser capaz de resolver qualquer problema matemático com a precisão necessária para este problema específico, sem desperdiçar memória, ou recursos computacionais." Anônimo.

Por acaso a amável leitora lembra que eu falei da relação de um para um entre um número real e a sua representação em memória? A norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) padronizou a representação binária de números de ponto flutuante e resolveu todos os problemas de compatibilidade entre hardware, software e mesmo entre soluções diferentes que existiam garantindo explicitamente a existência desta relação biunívoca entre o número decimal e o número binário que será usado para armazená-lo em memória. Assim, todas as máquinas, e todos os softwares, entenderam o mesmo conjunto de bits, da mesma forma.

A norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) não é a única forma de armazenar números reais, talvez não seja sequer a melhor forma, mas é de longe a mais utilizada. Com esta norma embaixo do braço, saberemos como representar uma _faixa significativa_ de números reais podendo determinar exatamente a precisão máxima possível para cada valor representado, mesmo em binário e, principalmente, conheceremos todos os problemas inerentes a esta representação. E existem problemas. Afinal, números decimais reais e infinitos serão mapeados em um universo binário, inteiro e finito. O que poderia dar errado?

Quase esqueci! A expressão _faixa significativa_ que usei acima é para destacar que a norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) não permite a representação de todo e qualquer número real. Temos um número infinito de valores na base $10$ representados em um número finito de valores na base $2$.

## E os binários entram na dança

Para trabalhar com qualquer valor em um computador, precisamos converter os números reais na base $10$ que usamos diariamente para base $2$ que os computadores usam. Armazenar estes números, realizar cálculos com os binários armazenados e, finalmente converter estes valores para base $10$ de forma que seja possível ao pobre ser humano entender a informação resultante do processo computacional. É neste vai e volta que os limites da norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) são testados e, não raramente, causam alguns espantos e muitos problemas.

Tomemos, por exemplo o número decimal $0,1_{10}$. Usando o [Decimal to Floating-Point Converter](https://www.exploringbinary.com/floating-point-converter/) para poupar tempo, e precisão dupla, já explico isso, podemos ver que:

$$0,1_{10} = (0.0001100110011001100110011001100110011001100110011001101)_2$$

Ou seja, nosso $0,1_{10}$ será guardado em memória a partir de:

$$(0.0001100110011001100110011001100110011001100110011001101)_2$$

Um belo de um número binário que, será armazenado segundo as regras da norma IEEE 754 e em algum momento será convertido para decimal resultando em:

$$(0.1000000000000000055511151231257827021181583404541015625)_{10}$$

Eita! Virou outra coisa. Uma coisa bem diferente. Eis porquê em Python, acabamos encontrando coisas como:

```python
>0.1 * 3
>0.30000000000000004
```

Isto ocorre por que a conta que você realmente fez foi $0.1000000000000000055511151231257827021181583404541015625 \times 3$. Se não acreditar em mim, tente você mesmo, direto na linha de comando do Python ou em alguma célula do [Google Colab](https://colab.research.google.com/). Vai encontrar o mesmo erro. Talvez esta seja uma boa hora para se levantar, tomar um copo d'água e pensar sobre mudança de carreira. Ouvi falar que jornalismo, contabilidade, educação física, podem ser boas opções.

Muitas linguagens de programação, o Python, inclusive, conhecem um conjunto de valores onde erros deste tipo ocorrem e arredondam, ou truncam, o resultado para que você veja o resultado correto. Ou ainda, simplesmente limitam o que é exposto para outras operações, como se estivessem limitando a precisão do cálculo ou dos valores armazenados. Não é raro encontrar linguagens de programação que, por padrão, mostram apenas 3 casas depois da vírgula. Esta foi uma opção pouco criativa adotada por muitos compiladores e interpretadores que acaba criando mais problemas que soluções. Para ver um exemplo, use a fração $\frac{1}{10}$, ainda em Python e reproduza as seguintes operações:

```python
> 1 / 10
>0.1
```

Viu que lindo? Funciona direitinho. É bem isso que deveria acontecer. A matemática é linda! Os céticos devem experimentar algo um pouco mais complicado, ainda utilizando o Python:

```python
a = 1/10
print( a)
print ("{0:.20f}".format(a))
0.10000000000000000555
```

E não é que a coisa não é tão linda assim! A diferença entre estes dois exemplos está na saída. No último formatamos a visualização do resultado para forçar a exibição de mais casas decimais mostrando que o erro está lá. Você não está vendo este erro, o interpretador vai tentar não permitir que este erro se propague, mas ele está lá. E, vai dar problema. E como tudo que causa problemas vai acontecer no pior momento possível.

> "Os interpretadores e compiladores são desenvolvidos por seres humanos, tão confiáveis quanto pescadores e caçadores. Não acredite em histórias de pescaria, de caçada ou de compilação" Frank de Alcantara.

_Isto não é uma exclusividade do Python_, a maioria das linguagens de programação, sofre de problemas semelhantes em maior ou menor número. Mesmo que os compiladores e interpretadores se esforcem para não permitir a propagação deste erro se você fizer uma operação com o valor $0.1$, que a linguagem lhe mostra com algum outro valor que exija, digamos $20$ dígitos depois da vírgula o erro estará lá.

Vamos abandonar o computador por um momento. Pegue a sua calculadora e divida um por três e veja o último dígito da tela, se for um seis, sua calculadora trunca a resposta, se for um sete, sua calculadora arredonda. A diferença está na precisão do número representado: 

1. **Truncar**: elimina todas as casas decimais após um certo ponto, sem considerar o valor das casas decimais que estão sendo removidas.
   - Exemplo:
     - Número original: 5.789
     - Número truncado (até uma casa decimal): 5.7

2. **Arredondar**: aumenta ou diminui o último dígito retido, dependendo do valor do próximo dígito que está sendo removido.
   - Exemplo:
     - Número original: 5.789
     - Número arredondado (até uma casa decimal): 5.8

Sistemas, mesmo que sejam simples calculadoras, que arredondam são mais precisos. Existem várias técnicas de arredondamento que não serão objeto deste artigo. Mesmo que eu tenha ficado tentado.

Volte um pouquinho e reveja o que aconteceu, no Python, quando operamos $0.1 * 3$. A leitora deve observar que, neste caso, os dois operandos estão limitados e são exatos. O erro ocorre por que a conversão de $0.1_{10}$ para binário não é exata e somos forçados a parar em algum ponto e, ou truncamos ou arredondamos o valor. Digamos que paramos em: $0.0001100110011001101_2$. Se fizermos isso e convertemos novamente para o decimal o $0.1$ será convertido em $0.1000003814697265625$. E lá temos um baita de um erro. Se a conversão for feita usando os padrões impostos pela [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) os valores ficam um pouco diferentes, o valor $0.1$ será armazenado em um sistema usando a [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) como:

1. em precisão simples:

   $$00111101 11001100 11001100 11001101_2$$

2. em precisão dupla:

   $$00111111 10111001 10011001 10011001 10011001 10011001 10011001 10011010_2$$

Que quando convertidos novamente para binário, precisão simples, representará o número $0.100000001490116119385$ isso implica em um erro $256$ vezes menor que o erro que obtemos com a conversão manual e os poucos bits que usamos. Em precisão dupla este valor vai para $0.100000000000000005551_2$ com um erro ainda menor. Nada mal! 

Vamos ver se entendemos como esta conversão pode ser realizada usando o $0,1$. Mas antes divirta-se um pouco vendo o resultado que obtemos graças a [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) para: $0,2$; $0,4$ e $0,8$ usando o excelente [Float Point Exposed](https://float.exposed). Como disse antes: tem pressa não!

## Entendendo a IEEE 754

A norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) especifica 5 formatos binários: meia precisão - $16$ bits; precisão simples - $32$ bits; precisão dupla - $64$ bits; precisão quadrupla - $128$ bits e precisão óctupla - $256$ bits. Se olhar com cuidado, exitem algumas variações em torno deste tema. Além dos formatos de precisão simples e dupla apresentados aqui, a norma IEEE 754 também define formatos de menor precisão (meia precisão com $16$ bits) e maior precisão (quadrupla com $128$ bits e octupla com $256$ bits). Há também algumas variações em relação à representação do infinito e do NaN. Contudo, por uma questão didática, neste artigo nos ateremos às duas representações de bits mais comumente utilizadas que são as precisões simples e dupla.

<img class="img-fluid" src="{{ site.baseurl }}/assets/images/ieee754.png" alt="mostra as distribuição de bits o padrão ieee 754">

Um valor real na base $10$, será convertido em binário e ocupará o espaço de $32$, ou $64$ bits, dependendo da precisão escolhida e das capacidades físicas da máquina que irá armazenar este dado. Nos dois casos, o primeiro bit, o bit mais significativo, será reservado para indicar o sinal do número armazenado. Quando encontramos o $1$ neste bit temos a representação de um valor negativo armazenado o zero no bit mais significativo indica um valor positivo. Os próximos $8$ bits, para a precisão simples ou $11$ bits para a precisão dupla, são reservados para o expoente que usaremos para a representação em ponto flutuante. Volto ao expoente já, já. Agora vamos dar uma olhada nos bits que restam além do sinal e do expoente, nestes bits armazenaremos a mantissa, a parte significativa do valor que estamos armazenando.

A terceira seção, que comporta $23$ bits em precisão simples e $52$ em precisão dupla é chamada de mantissa e contém o binário equivalente aos algoritmos significativos do número que vamos armazenar. A leitora deve ser lembrar que eu pedi para guardar o número $1,01011101 \times 2^5$, Lembra? A nossa mantissa, em precisão simples tem espaço para $23$ bits poderíamos, simplesmente, armazenar $10101110100000000000000$. E, neste ponto, temos que parar e pensar um pouco.

Na notação científica, como definimos anteriormente, não podemos ter um zero antes da vírgula. O mesmo deve ser considerado para a notação científica quando usamos números em binário. Com uma grande diferença: se o algarismo antes da vírgula não pode ser um zero ele obrigatoriamente será o $1$. Afinal, estamos falando de binário. Ou seja, **a mantissa não precisa armazenar o algarismo antes da vírgula**. Sendo assim, para armazenar a mantissa de $1,01011101 \times 2^5$ vamos utilizar apenas $01011101_2$ que resultará em $01011101000000000000000_2$ uma precisão maior graças ao zero a mais. A leitora tinha contado os zeros? Está claro que preenchemos os $32$ bits do mais significativo para o menos significativo por que estamos colocando algoritmos depois da vírgula?

A mantissa é simples e não há muito para explicar ou detalhar. A leitora, se estiver curiosa, pode complementar este conhecimento e ler sobre a relação entre casas decimais em binário e as casas decimais na base dois [neste link](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470124604.app15). Posso apenas adiantar que esta relação tende a $log_2(10) \equiv 3.32$. Isto implica na necessidade de aproximadamente $3.32$ vezes mais algoritmos em binário que em decimal para representar a mesma precisão.

Esta foi a parte fácil, a leitora deve ser preparar para os expoentes. Só para lembrar, temos $8$ bits em precisão simples e $11$ bit em precisão dupla.

**Considerando a precisão simples**, entre os $8$ bits reservados para a representação do expoente não existe um bit que seja específico para indicar expoentes negativos. Em vez disso, os valores são representados neste espaço de $8$ bits em uma notação chamada de **excess-127 ou bias**. Nesta notação, utilizamos um número inteiro de $8$ bits cujo valor sem sinal é representado por $M-127$ como expoente. Desta forma, O valor $01111111$ equivalente ao valor $127$ representa o expoente $0$ em decimal, o valor $01000000$ equivalente a $128$, representa o expoente $1$, enquanto o valor $01111110$ equivalente a $126$ representa o expoente $-1$ e por ai vamos. Em outras palavras, para representar o expoente $0$ armazenamos o valor binário $M=01111111$ equivalente ao $127$ e o expoente será dado por $$M$$ subtraído do valor $127$, ou seja $0$. Usando esta técnica **excess-127 ou bias** teremos uma faixa de expoentes que variam $2^{-126}$ e $2^{128}$ para a precisão simples. Parece complicado e é mesmo.

**No caso da precisão dupla** o raciocínio é exatamente o mesmo exceto que o espaço é de $11$ bits e o _bias_ é de $1023 (excess-1023)$. Com $11$ bits conseguimos representar valores entre $0$ e $2047$. Neste caso, o $M=1023$ irá representar o valor $0$. Com a precisão dupla poderemos representar expoentes entre $-1022$ e $1023$. Em resumo:

1. em precisão simples um expoente estará na faixa entre $-126$ e $127$ com um _bias_ de $127$ o que permitirá o uso de algorítmos entre $1$ e $254$, os valores $0$ e $255$ são reservados para representações especiais.
2. em precisão dupla um expoente estará na faixa entre $-1022$ e $1023$ com um _bias_ de $1023$ o que permitirá o uso de valores entre $1$ e $2046$, os valores $0$ e $2047$ são reservados para representações especiais.

Parafraseando um dos personagens do filme [Bolt](<https://pt.wikipedia.org/wiki/Bolt_(2008)>), a leitora deve colocar um _pin_ na frase: **são reservados para representações especiais** nós vamos voltar a isso mais trade. Por enquanto vamos voltar ao $0,1_{10}$. Este é valor numérico que mais irrita todo mundo que estuda este tópico. Deveria ser simples é acaba sendo muito complexo.

## De decimal para IEEE 754 na unha

A leitora terá que me dar um desconto, vou fazer em precisão simples. Haja zeros! E, por favor, preste atenção só vou fazer uma vez. :)

Antes de qualquer relação com a norma IEEE 754, vamos converter $0,1_{10}$ para binário. Começamos pela parte inteira deste número. Para isso vamos dividir o número inteiro repetidamente por dois, armazenar cada resto e parar quando o resultado da divisão, o quociente, for igual a zero e usar todos os restos para representar o número binário:

$$0 \div 2 = 0 + 0 \therefore 0_{10} = 0_2$$

Esta parte foi fácil $0_{10}$ é igual a $0_2$.

Em seguida precisamos converter a parte fracionária do número $0,1$ multiplicando este algoritmo repetidamente por dois até que a parte fracionária, aquilo que fica depois da vírgula, seja igual a zero e já vamos separando a parte inteira, resultado da multiplicação da parte fracionária. Vamos armazenar a parte inteira enquanto estamos multiplicando por dois a parte fracionária do resultado de cada operação anterior. Ou seja, começamos com $0,1 \times 2 = 0,2$ temos $0$ parte inteira do resultado da multiplicação e $0,2$ parte fracionária do resultado que vamos representar por $0,1 \times 2 = 0 + 0,2$ e assim sucessivamente:

<table class="table table-striped">
  <tr>
    <td>1. </td><td>$0,1 × 2 = 0 + 0,2$ </td>
    <td style="text-align: right;">13. </td><td>$0,2 × 2 = 0 + 0,4$ </td>
  </tr>
  <tr>
    <td>2. </td><td>$0,2 × 2 = 0 + 0,4$ </td>
    <td style="text-align: right;">14. </td><td>$0,4 × 2 = 0 + 0,8$ </td>
  </tr>
  <tr>
    <td>3. </td><td>$0,4 × 2 = 0 + 0,8$ </td>
    <td style="text-align: right;">15. </td><td>$0,8 × 2 = 1 + 0,6$ </td>
  </tr>
  <tr> 
    <td>4. </td><td>$0,8 × 2 = 1 + 0,6$ </td>
    <td style="text-align: right;">16. </td><td>$0,6 × 2 = 1 + 0,2$ </td>
  </tr>
  <tr>
    <td>5. </td><td>$0,6 × 2 = 1 + 0,2$ </td>
    <td style="text-align: right;">17. </td><td>$0,4 × 2 = 0 + 0,8$ </td>
  </tr>
  <tr>
    <td>6. </td><td>$0,4 × 2 = 0 + 0,8$ </td>
    <td style="text-align: right;">18. </td><td>$0,8 × 2 = 1 + 0,6$ </td>
  </tr>
  <tr> 
    <td>7. </td><td>$0,8 × 2 = 1 + 0,6$ </td>
    <td style="text-align: right;">19. </td><td>$0,6 × 2 = 1 + 0,2$ </td>
  </tr>
  <tr> 
    <td>8. </td><td>$0,6 × 2 = 1 + 0,2$ </td>
    <td style="text-align: right;">20. </td><td>$0,2 × 2 = 0 + 0,4$ </td>
  </tr>
  <tr>
    <td>9. </td><td>$0,2 × 2 = 0 + 0,4$ </td>
    <td style="text-align: right;">21. </td><td>$0,4 × 2 = 0 + 0,8$ </td>
  </tr>
  <tr>
    <td>10. </td><td>$0,4 × 2 = 0 + 0,8$ </td>
    <td style="text-align: right;">22. </td><td>$0,8 × 2 = 1 + 0,6$ </td>
  </tr>
  <tr>
    <td>11. </td><td>$0,8 × 2 = 1 + 0,6$ </td>
    <td style="text-align: right;">23. </td><td>$0,6 × 2 = 1 + 0,2$ </td>
  </tr>
  <tr>
    <td>12. </td><td>$0,6 × 2 = 1 + 0,2$ </td>
    <td style="text-align: right;">24. </td><td>$0,2 × 2 = 0 + 0,4$ </td>
  </tr>
  <tr>
    <td><td> </td> </td>
    <td style="text-align: right;">25. </td><td>$0,4 × 2 = 0 + 0,8$ </td>
  </tr>
</table>

Podemos continuar e não vamos conseguir encontrar um resultado de multiplicação cuja parte fracionária seja igual a $0$, contudo como na mantissa, em precisão simples, cabem 23 bits, acho que já chegamos a um nível suficiente de precisão. Precisamos agora ordenar todas as partes inteiras que encontramos para formar nosso binário:

$$0,1_{10} = 0,000110011001100110011001100_2$$

Resta normalizar este número. A leitora deve lembrar que a representação normal, não permite o $0$ como algarismo inteiro (antes da vírgula). O primeiro $1$ encontra-se na quarta posição logo:

$$0,0001 1001 1001 1001 1001 1001 100_2 \\ = 1.1001 1001 1001 1001 1001 100_2 \times 2^{-4}$$

Precisamos agora normalizar nosso expoente. Como estamos trabalhando com precisão simples usaremos $127$ como _bias_. Como temos $-4$ teremos $(-4+127) = 123$ que precisa ser convertido para binário. Logo nosso expoente será $01111011$.

Até agora temos o sinal do número, $0$ e o expoente $01111011$ resta-nos terminar de trabalhar a mantissa. Podemos remover a parte inteira já que em binário esta será sempre $1$ devido ao $0$ não ser permitido. Feito isso, precisamos ajustar seu comprimento para $23$ bits e, temos nossa mantissa: $10011001100110011001100$. Linda! E resumo temos:

<table class="table table-striped"> 
  <tr>
    <th style="text-align:center !important;"> Elemento </th>
    <th style="text-align:center !important;"> Valor </th>
  </tr>
  <tbody>
    <tr><td style="text-align:center !important;">Sinal</td><td>$(+) = 1$</td></tr>
    <tr><td style="text-align:center !important;">Expoente</td><td>$(123_{10}) = 01111011_2$</td></tr>
    <tr><td style="text-align:center !important;">Mantissa</td><td>$10011001100110011001100$</td></tr>
    <tr><td style="text-align:right !important;">Total</td><td>$32 \space bits$</td></tr>
  </tbody>
</table>

## Os valores especiais

A leitora deve lembrar da expressão que pedi que colocasse um pin: **são reservados para representações especiais**. Está na hora de tocar neste assunto delicado. A verdade é que não utilizamos a IEEE 754 apenas para números propriamente ditos, utilizamos para representar todos os valores possíveis de representação em um ambiente computacional que sejam relacionados a aritmética dos números reais. Isto quer dizer que temos que armazenar o zero, o infinito e valores que não são numéricos, os famosos **NAN**, abreviação da expressão em inglês _Not A Number_ que em tradução livre significa **não é um número**. A forma como armazenamos estes valores especiais estão sintetizados na tabela a seguir:

<table class="table table-striped">
<tr>
  <th colspan="2" style="text-align:center !important;">Precisão Simples</th>
  <th colspan="2" style="text-align:center !important;">Precisão Dupla</th>
  <th></th>
</tr>
<tbody>
<tr>
  <td style="text-align:center !important;">Expoente</td>
  <td style="text-align:center !important;">Mantissa</td>
  <td style="text-align:center !important;">Expoente</td>
  <td style="text-align:center !important;">Mantissa</td>
  <td style="text-align:center !important;">Valor Representado</td>
</tr>
<tr>
  <td> $0$ </td>
  <td> $0$ </td>
  <td> $0$ </td>
  <td> $0$ </td>
  <td> $\pm 0$ </td>
</tr>
<tr>
  <td> $0$ </td>
  <td> $ \neq 0$ </td>
  <td> $0$ </td>
  <td> $ \neq 0$ </td>
  <td> $\pm \space Número \space Subnormal$</td>
</tr>
<tr>
  <td> $1-254$ </td>
  <td> $Qualquer \space valor$ </td>
  <td> $1-2046$ </td>
  <td> $Qualquer \space valor$ </td>
  <td> $\pm \space Número \space Normal $</td>
</tr>
<tr>
  <td> $255$ </td>
  <td> $0$ </td>
  <td> $2047$ </td>
  <td> $0$ </td>
  <td> $\pm \space Infinito$</td>
</tr>
<tr>
<td> $255$ </td>
<td> $\neq 0$ </td>
<td> $2047$ </td>
<td> $\neq 0$ </td>
<td> $NaN \space (Not \space a \space Number)$</td>
</tr>
</tbody></table>

Resta-nos entender o que estes valores representam e seu impacto na computação.

### Números subnormais

Para a IEEE 754 normal é tudo que vimos anteriormente, todos os valores que podem ser representados usando as regras de sinal, expoente e mantissa de forma normalizada que a amável leitora teve a paciência de estudar junto comigo. Subnormal, ou não normalizado, é o termo que empregamos para indicar valores nos quais o campo expoente é preenchido com zeros. Se seguirmos a regra, para representar o algarismo $0$ o expoente deveria ser o $-127$. Contudo, para este caso, onde todo o campo expoente é preenchido com $00000000$ o expoente será $-126$. Neste caso especial, a mantissa não terá que seguir a obrigatoriedade de ter sempre o número $1$ como parte inteira. Não estamos falando de valores normalizados então o primeiro bit pode ser $0$ ou $1$. Estes números foram especialmente criados para aumentar a precisão na representação de números que estão no intervalo entre $0$ e $1$ melhorando a representação do conjunto dos números reais nesta faixa.

A leitora há de me perdoar novamente, a expressão subnormal é típica da norma IEEE 854 e não da IEEE 754, mas tomei a liberdade de usar esta expressão aqui por causa da melhor tradução.

### Zero

Observe que a definição de zero na norma IEEE 754 usa apenas o expoente e a mantissa e não altera nada no bit que é utilizado para indicar o sinal de um número. A consequência disto é que temos dois números binários diferentes um para $+0$ e outro para $-0$. A leitora deve pensar no zero como sendo apenas outro número subnormal que, neste caso acontece quando o expoente é $0$ e a mantissa é $0$. Sinalizar o zero não faz sentido matematicamente e tanto o $+0$ quanto o $-0$ representam o mesmo valor. Por outro lado, faz muita diferença do ponto de vista computacional e é preciso atenção para entender estas diferenças.

### Infinito

Outro caso especial do campo de exponentes é representado pelo valor $11111111$. Se o expoente for composto de $8$ algarismos $1$ e a mantissa for totalmente preenchida como $0$, então o valor representado será o infinito. Acompanhando o zero, o infinito pode ser negativo, ou positivo.

Neste caso, faz sentido matematicamente. Ou quase faz sentido. Não, não faz sentido nenhum! Não espere, faz sim! Droga infinito é complicado. A verdade é que ainda existem muitas controvérsias sobre os conceitos de infinito, mesmo os matemáticos não tem consenso sobre isso, a norma IEEE 754 com o $\pm Infinito$ atende ao entendimento médio do que representa o infinito.

Se você está usando uma linguagem de programação que segue a norma IEEE 754, você notará algo interessante ao calcular o inverso de zero. Se fizer o cálculo com $-0$, o resultado será $-\infty$. Se fizer o cálculo com $+0$, o resultado será $+\infty$.

Do ponto de vista estritamente matemático, isso não é exatamente correto. Matematicamente, a divisão de qualquer número por zero não é definida - diz-se que ela tende ao infinito, mas não é igual ao infinito.

O que a norma IEEE 754 está fazendo aqui é uma espécie de compromisso prático. Ela nos dá uma indicação da direção em que o resultado está indo (para o infinito positivo ou negativo), mesmo que isso não seja uma representação exata do que acontece na matemática pura. Assim, em termos de programação, obtemos uma resposta útil, mesmo que essa resposta não seja rigorosamente precisa do ponto de vista matemático.

### NaN (Not a Number)

O conceito de **NaN** foi criado para representar valores, principalmente resultados, que não correspondem a um dos números reais que podem ser representados em binário segundo a norma IEEE 754. Neste caso o expoente será completamente preenchido como $1$ e a mantissa será preenchida com qualquer valor desde que este valor não seja composto de todos os algarismos com o valor $0$. O bit relativo ao sinal não causa efeito no NaN. No entanto, existem duas categorias de NaN: QNaN _(Quiet NaN)_ e SNaN _(Signalling NaN)_.

O primeiro caso QNaN, _(Quiet NaN)_, ocorre quando o bit mais significativo da mantissa é $1_2$. O QNaN se propaga na maior parte das operações aritméticas e é utilizado para indicar que o resultado de uma determinada operação não é matematicamente definido. já o SNaN, _(Signalling NaN)_, que ocorre quando o bit mais significativo da mantissa é $0_2$ é utilizado para sinalizar alguma exceção como o uso de variáveis não inicializadas. Podemos sintetizar estes conceitos memorizando que QNaN indica operações indeterminadas enquanto SNaN indica operações inválidas.

  <table class="table table-striped">
        <tr style="text-align: center;">
          <th>Operação</th>
          <th>Resultado</th>
        </tr>
    <tbody>
        <tr style="text-align: center;">
          <td>$(Número) \div (\pm \infty)$ </td>
          <td> $0$ </td>
        </tr>
        <tr style="text-align: center;">
            <td>$(\pm \infty) \times (\pm \infty)$</td>
            <td>$\pm \infty$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(\pm \neq 0) \div (\pm 0)$</td>
          <td>$\pm \infty$</td>
        </tr>
        <tr style="text-align: center;">
            <td>$(\pm Número) \times (\pm \infty)$</td>
            <td>$\pm \infty$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(\infty) + (\infty)$</td>
          <td>$+\infty$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(\infty) - (-\infty)$</td>
          <td>$+\infty$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(-\infty) + (-\infty)$</td>
          <td>$-\infty$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(-\infty) - (\infty)$</td>
          <td>$-\infty$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(\infty) - (\infty)$</td>
          <td>$NaN$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(-\infty) + (\infty)$</td>
          <td>$NaN$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(\pm 0) \div (\pm 0)$</td>
          <td>$NaN$</td>
        </tr>
        <tr style="text-align: center;">
           <td>$(\pm \infty) \div (\pm \infty)$</td>
           <td>$NaN$</td>
        </tr>
        <tr style="text-align: center;">
          <td>$(\pm \infty) \times (0)$</td>
          <td>$NaN$</td>
        </tr>
        <tr style="text-align: center;">
            <td>$(NaN) == (NaN)$</td>
            <td>$false$</td>
        </tr>
    </tbody>
  </table>

Antes de chamar os aldeões e começar as fogueiras a amável leitora precisa levar em consideração as intensões que suportam a norma IEEE 754. Originalmente o objetivo era criar um ambiente padrão para a troca de números em ponto flutuante entre máquinas e softwares. Resolvendo milhares de problemas de compatibilidade que impediam o progresso da computação. E só.

No esforço que criar uma camada de compatibilidade, foi criado todo um padrão que permite operar com estes números em um grau de precisão aceitável para a imensa maioria das operações computacionais.

Durante a criação da norma, ninguém se preocupou muito que valores especiais como $\pm Infinito$ ou $NaN$ seriam usados para qualquer coisa diferente de criar interrupções e sinalizar erros. Foi o tempo que apresentou situações interessantes que precisaram de detalhamento da norma. Notadamente quando passamos a exigir dos nossos programas comportamentos numericamente corretos para a resolução de problemas complexos.

O $-0$ e o $+0$ representam exatamente o mesmo valor mas são diferentes $-0 \neq +0$ o que implica que em alguns casos, nos quais, mesmo que $x=y$ eventualmente podemos ter que $\frac{1}{x} \neq \frac{1}{y}$ para isso basta que algum momento durante o processo de computação $x=-0$ e $y=+0$ o que já é suficiente para criar uma grande quantidade de problemas. Antes de achar que isso é muito difícil lembre-se, por favor, que existe um número próximo do infinito, só para ficar no dialeto que estamos usando, de funções que cruzam os eixos de um plano cartesiano. Um ponto antes estas funções estarão em $-0$ e um ponto depois em $+0$. Se tratarmos a existência do $\pm 0$ como interrupção ou alerta, podemos gerir estas ocorrências eficientemente e manter a integridade da matemática em nossos programas. Na matemática $+0$ e $-0$ são tratados da mesma forma. Se formos observar cuidadosamente os cálculo e utilizar estes dois valores de zero de forma diferente então, teremos que prestar muita atenção nas equações que usaremos em nossos programas.

O infinito é outro problema. Pobres de nós! Estes conceitos foram inseridos na norma para permitir a concordância com a ideia que o infinito é uma quantidade, maior que qualquer quantidade possivelmente representada e atende a Teoria Axiomática de [Zermelo–Fraenkel](https://en.wikipedia.org/wiki/Zermelo%E2%80%93Fraenkel_set_theory). Isto é importante porque hoje, esta é a teoria axiomática da teoria dos conjuntos que suporta toda a matemática. Vamos deixar Zermelo–Fraenkel para um outro artigo já que este conhecimento não faz parte do cabedal de conhecimentos do programador mediano. Basta lembrar que as operações aritméticas são coerentes e que, na maior parte das linguagens é possível trabalhar isso como um alerta.

Por fim, temos o $NaN$ este valor indica uma operação inválida, como $0 \div 0$ ou $\sqrt(-1)$. Este valor será propagado ao longo da computação, assim que surgir como resultado, permitindo que a maioria das operações que resultem em $NaN$, ou usem este valor como operador, disparem algum tipo de interrupção, ou alerta, que indique que estamos trabalhando fora dos limites da matemática e, muitas vezes, da lógica. Novamente, os problemas ocorrem graças as decisões que tomamos quando criamos uma linguagem de programação. Hoje não é raro encontrar programas onde o valor $$NaN$$ seja utilizado como um valor qualquer inclusive em operações de comparação. Pobres de nós!

> Esta aritmética foi criada para que qualquer programador, mesmo o mais ignorante, fosse avisado de que algo estava fora do normal e não para que os meandros da teoria dos números fossem explorados. [William Kahan](https://amturing.acm.org/award_winners/kahan_1023746.cfm).

A leitora deve fazer um esforço para me compreender nesta última citação. [Não é uma citação literal](https://people.eecs.berkeley.edu/~wkahan/ieee754status/why-ieee.pdf), trata-se de uma paráfrase de um dos criadores da norma IEEE 754. Entendendo a intensão que suporta o ato, entendemos as consequências deste ato. A norma permite o uso de valores de forma algebricamente correta. E isto deveria bastar. Até que a gente encontra linguagens como o javascript.

```javascript
> typeof NaN
> "number"

> NaN = NaN
> false;
```

As duas operações estão perfeitamente corretas segundo a norma, mas não fazem nenhum sentido, pelo menos não para quem ignora a norma. Sim, realmente $NaN$ é um número e sim, $NaN = NaN$ é falso. Em [Javascript: the weird parts](https://charlieharvey.org.uk/page/javascript_the_weird_parts) Charlie Harvey explora muitas das incongruências encontradas no javascript apenas porque os interpretadores seguem rigidamente as normas sem atentar para as razões da existência destas normas.

Aqui eu usei exemplos do Python e do Javascript porque são mais fáceis de testar. Nenhuma linguagem de programação imperativa está livre destes problemas. Se quiser dar uma olhada em C++, no Windows, John D. Cook em [IEEE floating-point exceptions in C++](https://www.johndcook.com/blog/IEEE_exceptions_in_cpp/) mostra como fazer isso.

> Uma coisa deve ficar para sempre: não use pontos flutuantes para dinheiro e nunca use _float_ se o _double_ estiver disponível. Só use float se estiver escrevendo programas em ambientes muito, muito, muito limitados em memória.

Certa vez [Joel Spolsky](https://www.joelonsoftware.com/) criou o termo _leaky abstraction_ que eu aqui, em tradução livre vou chamar de **abstração fraca**. A computação é toda baseada em metáforas e abstrações. Uma abstração forte é aquela em que você usa uma ferramenta sem nunca ter que abrir e ver o que há lá dentro. Uma abstração fraca é aquela em que você tem que abrir a ferramenta antes de usar. **Pontos flutuantes são abstrações fracas**. E, apesar de todas as linguagens de programação que eu conheço usarem esta norma, a leitora não está obrigada a usar esta norma nos seus programas, mas isto é assunto para outro artigo.

# Referências

1. [Binary System](https://binary-system.base-conversion.ro/real-number-converted-from-decimal-system-to-32bit-single-precision-IEEE754-binary-floating-point.php?decimal_number_base_ten=0.2). Acessado em 17 de julho de 2023.

2. [Floating Point](https://users.cs.duke.edu/~raw/cps104/TWFNotes/floating.html). Acessado em 17 de julho de 2023.

3. [Floating Point Arithmetic: Issues and Limitations](https://docs.python.org/3/tutorial/floatingpoint.html). Acessado em 17 de julho de 2023.

4. [Floating Point Numbers](https://floating-point-gui.de/formats/fp/). Acessado em 17 de julho de 2023.

5. [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754#Rounding_rules). Acessado em 17 de julho de 2023.

6. [IEEE floating-point exceptions in C++](https://www.johndcook.com/blog/IEEE_exceptions_in_cpp/). Acessado em 17 de julho de 2023.

7. [Javascript: the weird parts](https://charlieharvey.org.uk/page/javascript_the_weird_parts). Acessado em 17 de julho de 2023.

8. [Taming Floating Point Error](https://www.johnbcoughlin.com/posts/floating-point-axiom/). Acessado em 17 de julho de 2023.

9. [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html). Acessado em 17 de julho de 2023.

10. [Why do we need a floating-point arithmetic standard?](https://people.eecs.berkeley.edu/~wkahan/ieee754status/why-ieee.pdf). Acessado em 17 de julho de 2023.

11. [Zermelo–Fraenkel](https://en.wikipedia.org/wiki/Zermelo%E2%80%93Fraenkel_set_theory). Acessado em 17 de julho de 2023.
