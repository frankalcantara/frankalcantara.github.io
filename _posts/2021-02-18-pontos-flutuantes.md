---
layout: post
title: "Removendo um pouco da sujeira sobre os pontos flutuantes"
author: Frank
categories: [artigo]
tags: [interpretadores, estrutura de dados, modelagem]
image: assets/images/pontosflutu.jpg
featured: false
rating: 3.5
---

A memória disponível é limitada. Graças a isso, em algum momento temos que arredondar os números reais que queremos armazenar. A dificuldade está em decidir quanta informação precisamos armazenar e a forma e a forma deste armazenamento para garantir que os computadores sejam úteis resolvendo problemas reais. Uma das soluções encontradas foi o uso de números em ponto flutuante. Este artigo vai explicar o que é isso, como funciona, e os problemas que esta técnica causa.

<span>Foto de <a href="https://unsplash.com/@calliestorystreet?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Callie Morgan</a> on <a href="https://unsplash.com/s/photos/floating?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

Este problema de armazenamento não é exclusivo dos computadores, o caderno que a leitora usou para aprender a somar era limitado em quantidade de linhas por página e quantidade de páginas. Para que fique claro vamos começar observando os números decimais, os números na base $$10$$. Entre estes nos interessam os números reais. Específicamente, os números reais fracionários. Aqueles que possuem uma parte inteira, colocada antes da vírgula, e uma parte fracionária, depois da vírgula. Este termo fracionário tem origem na possibilidade de representar números reais na forma de operações com frações, formalmente, operações racionais. Por exemplo:

$$0,125 = \frac{1}{10}+\frac{2}{100}+\frac{5}{1000} = \frac{1}{10^1}+\frac{2}{10^2}+\frac{5}{10^3}$$

> "Deus criou os inteiros, todo o resto é trabalho dos homens." Leopold Kronecker

Infelizmente o homem erra e não é perfeito. Os números reais, raramente podem ser representados por uma operação entre frações de forma exata.

Tome, por exemplo, a razão $$\frac{1}{6}$$ e tente representá-la em números reais sem arredondar, ou truncar. A leitora verá, muito rapidamente, que isto não é possível e que em algum ponto teremos que arredondar, ou truncar. Obteremos, invariavelmente, algo como $$0,166667$$. O momento onde vamos parar de dividir e arredondar, ou truncar, determina a precisão que usaremos para representar este número e a precisão será, por sua vez, definida pelo uso que daremos a este número. Este uso é importante, define o modelo que usaremos para resolver um problema. E todos os problemas são diferentes.

Em uma estrada, a diferença de um centímetro que existe entre $$12,00m$$ e $$12,01m$$ provavelmente não fará qualquer diferença. Se estivermos construíndo um motor, por outro lado, um erro de $$1cm$$ será a diferença entre funcionar e explodir. Maximize este conceito imaginando-se no lugar de um um físico precisando utilizar a constante gravitacional. Neste caso, a leitora terá o desprazer de fazer contas com números como $$0.00000000006667_{10}$$.

Números reais não são adequados ao uso em computação. Nossos computadores são binários, trabalham só e somente só com números na base $$2$$. Estes computadores são máquinas limitadas pelo hardware e pelo software que executam. Sem pensar muito dá para perceber que existe um número infinito de números reais, representados por um número também infinito de precisões diferentes e que todo este universo deverá caber em um espaço restrito definido pela memória disponível e pelas regras da aritmética binária. Ao menos, os números binários são tão flexíveis quanto os números decimais. Ou qualquer outra base.

Os números na base dois podem ser representados por uma parte inteira e uma parte fracionária exatamente como fazemos com os números na base $$10$$. Vamos usar o número $$0.001$$ na base $$2$$ como exemplo. Este número pode ser representado por uma operação de frações na base adotada:

$$0,001 = \frac{0}{2}+\frac{0}{4}+\frac{1}{8} = \frac{0}{2^1}+\frac{0}{2^2}+\frac{1}{2^3}$$

Os números fracionários na base $$2$$A apresentam o mesmo problema que vimos na base $$10$$. A maioria dos números binários facionários, não pode ser representada de forma exata por uma operação de frações. um bom exemplo pode ser visto com a fração $$(\frac{1}{3})_{10}$$ que seria representada, em conversão direta para o binário, por $$(\frac{1}{10})_{2}$$. Esta representação, em binário, terá que ser arredondada, ou truncada. Definir este ponto, e a forma de armazenamento destes algarismos é, ainda hoje, uma necessidade com a qual temos que lidar.

Só para lembrar: nosso problema é encontrar uma forma de representar todo o universo de números reais, em base $$10$$, em um espaço limitado em base $$2$$. Se pensarmos em uma correspondência de um para um, todo e qualquer número real deve ser armazenado no espaço de dados definido por um e apenas um endereço de memória. A leitora precisa me permitir adiantar um pouco as coisas: isso é impossível.

<img class="img-fluid" src="{{ site.baseurl }}/assets/images/churchill1.jpg" alt="mostra as distribuição de bits o padrão ieee 754">

## E lá vem o homem com suas imperfeições

Rm 1985 o IEEE, _Institute of Electrical and Electronics Engineers_ publicou uma norma, a norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) cujo objetivo era padronizar uma única representação para números de ponto flutuante. Na época, os dois mais importantes fabricantes de hardware, Intel e Motorola, apoiaram e adotaram esta norma nas suas cpus o que contribuiu para a adoção que temos hoje. A norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) descreve como representar números em binário com precisão simples, $$32$$ bits, dupla, $$64$$ bits, quádrupla $$128$$ bits e óctupla $$256$$ principalmente mas não exclusivamente.

Antes de nos aprofundarmos nos meandros da representação binária de ponto flutuante, talvez seja uma boa ideia lembar o ensino médio. Foi lá que eu aprendi a notação científica. Pensando bem, hoje eu seria mais feliz se lá no Colégio Visconde de Mauá, alguém tivesse chamado a notação científica de representação de ponto flutuante. Em fim, segue a vida. Algumas coisas só entendemos muitos anos depois.

A verdade é que na base $$10$$ somos treinados a usar pontos flutuantes no que chamamos de **notação científica**, ou de notação de engenharia. Há uma pequena diferença entre a notação científica e a notação de engenharia que eu vou ignorar neste texto e me concentrar apenas na notação científica. Na notação científica temos a parte significativa do número, chamada de mantissa, e um expoente aplicado a uma base decima, $$10^e$$. Usando a base $$10$$ poderíamos ter:

<table class="table table-striped">
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
      <td>$$2,7$$</td>
      <td>$$4$$</td>
      <td>$$2,7 \times 10^4$$</td>
      <td>$$27.000$$</td>
    </tr>
    <tr>
      <td>$$-3.501$$</td>
      <td>$$2$$</td>
      <td>$$-3.501 \times 10^2$$</td>
      <td>$$-350.1$$</td>
    </tr>
    <tr>
      <td>$$7$$</td>
      <td>$$-3$$</td>
      <td>$$7 \times 10^{-3}$$</td>
      <td>$$0.007$$</td>
    </tr>
    <tr>
      <td>$$6,667$$</td>
      <td>$$-11$$</td>
      <td>$$6,667\times 10^{-11}$$</td>
      <td>$$0,00000000006667$$</td>
    </tr>
  </tbody>
</table>

Uma boa prática no uso da notação científica é deixar tantos algarismos significativos depois da vírgula quanto necessário para o cálculo específico que pretendemos realizar e apenas um algarismo antes da vírgula. Escolhemos a quantidade de números significativos de acordo com a aplicação. O $$\pi$$ é um bom exemplo de precisão relativa à aplicação.

Normalmente, um engenheiro civíl, ou eletricista, usa o $$\pi$$ como $$3.1416$$. Assim mesmo! Arredondando na última casa. Quatro algarismos significativos depois da vírgula resolvem a maior parte dos problemas no mundo sólido, real e visível. Este mesmo valor poderia ser representado por $$31,416 \times 10^{-1}$$, por mais estranho que pareça estaria matematicamente correto.

Leciono eletromagnetismo normalmente uso $$\pi = 3.1415926$$, igualmente arredondando a última casa mas com 7 algarismos significativos depois da vírgula, esta prática definiu meu modo de pensar. Achei muito estranho, e um tanto desnecessário, ter que usar $$\pi = 3.14159265359$$, truncado, sem nenhum arredondamento, com onze algarismos significativos depois da vírgula para resolver alguns problemas de cosmologia. Diga-se de passagem, nesta disciplina o único que achava isso estranho era eu. Pouco depois, descobri que o $$\pi$$ com onze algarismos significativos só era útil na primeira disciplina do curso. Em física de partículas, ou cosmologia, não é raro trabalhar com 30 dígitos de precisão. Neste ponto do nosso papo chegamos a duas regras muito importantes que a leitora não deve esquecer:

1. Cada problema tem a sua precisão particular e específica;
2. Não esqueça a primeira regra.

É preciso ficar claro que usei algumas notações pouco usuais para o valor de $$\pi$$. O fiz para que a leitora visse o ponto flutuar. Para que todos os escritos e todos os cálculos possam ser facilmente entendidos precisamos usar uma norma para representar estes números facionários então usamos a notação cientifica. Não é uma norma, no sentido estrito da palavra, é mais como um padrão que todos os matemáticos, físicos e cientistas foram usando ao longo do tempo e agora é ensinada em escolas e universidades em todo o planeta. Um padrão, pela praticidade e uso.

A norma da notação científica determina o uso de $$1$$, e somente $$1$$, algarismo antes da vírgula e permite definir os algarismos depois da virgula de acordo com a precisão necessária e, principalmente, a norma impõe que você nunca deve deixar penas o zero antes da vírgula. Adotando esta norma, $$3.1416$$ poderia ser representado por $$3.1416 \times 10^0$$, e estaria perfeitamente normalizado, ou por $$31,416\times 10^{-1}$$ que seria uma representação matematicamente válida, mas não normalizada. **Não estarão normalizados todos os números cuja parte inteira for $$0$$**.

Passou pela minha cabeça agora: está claro que a nomenclatura _ponto flutuante_ é importada do inglês? Se fosse em bom português seria vírgula flutuante. A gente vai falando, ou escrevendo, estas coisas, e nem se dá conta que não faz sentido no idioma de Ruben Braga.

A base não faz nenhuma diferença na norma da notação científica. Números binários podem ser representados nesta notação tão bem como números na base $$10$$ ou em qualquer outra base. A leitora pode, por exemplo usar o número $$43.625_{10}$$ que, convertido para binário seria $$101011,101_2$$ e representá-lo em notação científica como $$1,01011101 \times 2^5$$. Guarde este número, vamos precisar dele. Sério, guarde mesmo.

> "Idealmente, um computador deve ser capaz de resolver qualquer problema matemático com a precisão necessária para este problema específico, sem desperdiçar memória, ou recursos computacionais." Anônimo.

Por acaso a amável leitora lembra que eu falei da relação de um para um entre um número real e a sua representação? A norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) padronizou a representação binária de números de ponto flutuante e resolveu todos os problemas de compatibilidade entre hardware, software e mesmo entre soluções diferentes que existiam originalmente justamente garantindo esta relação biunívoca entre o número decimal e o número binário. Esta não é a única forma, talvez não seja a melhor forma, mas é de longe a mais utilizada. Com esta norma em mãos, sabemos como representar uma faixa significativa de números decimais e podemos determinar exatamente a precisão máxima possível para cada número, mesmo em binário e, principalmente, conhecemos os problemas inerentes a esta representação.

Quase esqueci! Não, a norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) não permite a representação de todo e qualquer número real. Temos um número infinito de valores na base $$10$$ representados em um número finito de valores na base $$2$$. Este é o momento em que a amável leitora deve se perguntar: o que poderia dar errado?

## E os binários entram na dança

Precisamos converter os números reais na base $$10$$ que usamos diariamente para base $$2$$ que os computadores usam, armazenar estes números, realizar cálculos com os binários e, finalmente converter estes valores para base $$10$$ de forma que seja possível ao pobre ser humano, entender a informação resultante do processo computacional. É neste vai e volta que os limites da norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) são testados e, não raramente, causam alguns espantos e muitos problemas.

Tomemos, por exemplo o número decimal $$0,1_{10}$$. Usando o [Decimal to Floating-Point Converter](https://www.exploringbinary.com/floating-point-converter/) para poupar tempo, e precisão dupla, já explico isso, podemos ver que:

$$0,1_{10} = (0.0001100110011001100110011001100110011001100110011001101)_2$$

Ou seja, nosso $$0,1_{10}$$ será guardado em memória a partir de:

$$(0.0001100110011001100110011001100110011001100110011001101)_2$$

Um belo de um número binário que, será armazenado segundo as regras da norma IEEE 754 e em algum momento será convertido para decimal resultando em:

$$(0.1000000000000000055511151231257827021181583404541015625)_{10}$$

Eita! Virou outra coisa. Uma coisa bem diferente. Eis porquê em Python, acabamos encontrando coisas como:

```python
>0.1 * 3
>0.30000000000000004
```

Isto ocorre por que a conta que você realmente fez foi $$0.1000000000000000055511151231257827021181583404541015625 \times 3$$. Se não acreditar em mim, tente você mesmo, direto na linha de comando do Python ou em alguma célula do [Google Colab](https://colab.research.google.com/). Vai encontrar o mesmo erro. Talvez esta seja uma boa hora para se levantar, tomar um copo d'água e pensar sobre jornalismo, contabilidade, educação física, ou qualquer outra opção de carreira que não envolva computação tão diretamente. Vai lá! Eu espero. Tem pressa não!

Muitas linguagens de programação, o Python, inclusive, conhecem um conjunto de valores onde erros deste tipo ocorrem e arredondam, ou truncam, o resultado para que você veja o resultado correto. Ou ainda, simplesmente limitam o que é exposto para outras operações, como se estivessem limitando a precisão do cálculo ou dos valores armazenados. Não é raro encontrar linguagens de programação que, por padrão, mostram apenas 3 casas depois da vírgula. Esta foi uma opção pouco criativa adotada por muitos compiladores e interpretadores que acaba criando mais problemas que soluções. Para ver um exemplo, use a fração $$frac{1}{10}$$, ainda em Python e reproduza as seguintes operações:

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

> "Os interpretadores e compiladores são desenvolvidos por seres humanos, tão confiáveis quanto pescadores e caçadores. Não acredite em histórias de pescaria, de caçada ou em compiladores" Frank de Alcantara.

_Isto não é uma exclusividade do Python_, a grande maioria das linguagens de programação, sofre de problemas semelhantes em maior ou menor número. Mesmo que os compiladores e interpretadores se esforcem para não permitir a propagação deste erro se você fizer uma operação com o valor $$0.1$$ que a linguagem lhe mostra com algum outro valor que exija, digamos 20 dígitos depois da vírgula o erro estará lá.

Volte um pouquinho e reveja o que aconteceu, no Python, quando operamos $$0.1 * 3$$. A leitora deve observar que, neste caso, os dois operandos estão limitados e são exatos. O erro ocorre por que a conversão de $$0.1_{10}$$ para binário não é exata e somos forçados a parar em algum ponto e, ou truncamos ou arredondamos o valor. Digamos que paramos em: $$0.0001100110011001101_2$$. Se fizermos isso e convertemos novamente para o decimal o $$0.1$$ será convertido em $$0.1000003814697265625$$. E lá temos um baita de um erro. Se a conversão for feita usando os padrões impostos pela IEEE 754 os valores ficam um pouco diferentes, o valor $$0.1$$ será armazenado em um sistema usando a IEEE 754 como:

1. em precisão simples:

   $$00111101 11001100 11001100 11001101_2$$

2. em precisão dupla:

   $$00111111 10111001 10011001 10011001 10011001 10011001 10011001 10011010_2$$

Que quando convertidos novamente para binário, precisão simples, representará o número $$0.100000001490116119385$$ isso implica em um erro 256 vezes menor que o erro que obtemos com a conversão manual e os poucos bits que usamos. Em precisão dupla este valor vai para $$0.100000000000000005551_2$$ com um erro ainda menor. Nada mal! Vamos ver se entendemos como esta conversão pode ser realizada usando o $$0,1$$. Mas antes divirta-se um pouco vendo o resultado que obtemos graças a IEEE 754 para: $$0,2$$; $$0,4$$ e $$0,8$$ usando o excelente [Float Point Exposed](https://float.exposed). Como disse antes: tem pressa não!

## Entendendo a IEEE 754

A norma IEEE 754 especifica 5 formatos binários: meia precisão - $$16$$ bits; precisão simples - $$32$$ bits; precisão dupla - $$64$$ bits; precisão quadrupla - $$128$$ bits e precisão óctupla - $$256$$ bits. Se olhar com cuidado, tem algumas variações em torno deste tema, contudo a leitora há de me perdoar por que vamos nos limitar as duas estruturas de bits mais comuns da IEEE 754, à saber:

<img class="img-fluid" src="{{ site.baseurl }}/assets/images/ieee754.png" alt="mostra as distribuição de bits o padrão ieee 754">

Um valor real, na base $$10$$ será convertido em binário e ocupará o espaço de $$32$$, ou $$64$$ bits dependendo da precisão escolhida e das capacidades físicas da máquina que irá armazenar este dado. Nos dois casos, o primeiro bit, o bit mais significativo, será reservado para indicar o sinal do número armazenado. Quando encontramos o $1$ neste bit temos a representação de um valor negativo armazenado o zero no bit mais significativo indica um valor positivo. Os próximos $$8$$ bits, para a precisão simples ou $$11$$ bits para a precisão dupla, são reservados para o expoente que usaremos para a representação em ponto flutuante. Volto ao expoente já, já. Agora vamos dar uma olhada nos bits que restam além do sinal e do expoente, nestes bits armazenaremos a mantissa, a parte significativa do valor que estamos armazenando.

A terceira seção, que comporta $$23$$ bits em precisão simples e $$52$$ em precisão dupla é chamada de mantissa e contém o binário equivalente aos algoritmos significativos do número que vamos armazenar. A leitora deve ser lembrar que eu pedi para guardar o número $$1,01011101 \times 2^5$$, Lembra? A nossa mantissa, em precisão simples tem espaço para $$23$$ bits poderíamos, simplesmente, armazenar $$10101110100000000000000$$. E, neste ponto, temos que parar e pensar um pouco.

Na notação científica, como definimos anteriormente, não podemos ter um zero antes da vírgula. O mesmo deve ser considerado para a notação científica quando usamos números em binário. Com uma grande diferença: se o algarismo antes da vírgula não pode ser um zero ele obrigatoriamente será o $$1$$. Afinal, estamos falando de binário. Ou seja, **a mantissa não precisa armazenar o algarismo antes da vírgula**. Sendo assim, para armazenar a mantissa de $$1,01011101 \times 2^5$$ vamos utilizar apenas $$01011101_2$$ que resultará em $$01011101000000000000000_2$$ uma precisão maior graças ao zero a mais. A leitora tinha contado os zeros? Está claro que preenchemos os $$32$$ bits do mais significativo para o menos significativo por que estamos colocando algoritmos depois da vírgula?

A mantissa é simples e não há muito para explicar ou detalhar. A leitora, se estiver curiosa, pode complementar este conhecimento e ler sobre a relação entre casas decimais em binário e as casas decimais na base dois [neste link](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470124604.app15). Posso apenas adiantar que esta relação tende a $$log_2(10) \equiv 3.32$$. Isto implica na necessidade de aproximadamente $$3.32$$ vezes mais algoritmos em binário que em decimal para representar a mesma precisão.

Esta foi a parte fácil, a leitora deve ser preparar para os expoentes. Só para lembrar, temos $$8$$ bits em precisão simples e $$11$$ bit em precisão dupla.

**Considerando a precisão simples**, entre os $$8$$ bits reservados para a representação do expoente não existe um bit que seja específico para indicar expoentes negativos. Em vez disso, os valores são representados neste espaço de $$8$$ bits em uma notação chamada de **excess-127 ou bias**. Nesta notação, utilizamos um número inteiro de $$8$$ bits cujo valor sem sinal é representado por $$M-127$$ como expoente. Desta forma, O valor $$01111111_2$$ equivalente ao valor $$127_{10}$$ representa o expoente $$0_{10}$$, o valor $$01000000_2$$ equivalente a $$128_{10}$$, representa o expoente $$1_{10}$$, enquanto o valor $$01111110_2$$ equivalente a $$126_{10}$$ representa o expoente $$-1_{10}$$ e por ai vamos. Em outras palavras, para representar o expoente $$0$$ armazenamos o valor binário $$M=01111111_2$$ equivalente ao $$127_{10}$$ e o expoente será dado por $$M$$ subtraído do valor $$127_{10}$$, ou seja $$0$$. Usando esta técnica **excess-127 ou bias** teremos uma faixa de expoentes que variam $$2^{-126}$$ e $$2^{128}$$ para a precisão simples. Parece complicado e é mesmo.

**No caso da precisão dupla** o raciocínio é exatamente o mesmo exceto que o espaço é de $$11$$ bits e o _bias_ é de 1023 (excess-1023). Com $$11$$ bits conseguimos representar valores entre $$0$$ e $$2047$$. Neste caso, o $$M=1023$$ irá representar o valor $$0$$. Com a precisão dupla poderemos representar expoentes entre $$-1022$$ e $$1023$$. Em resumo:

1. em precisão simples um expoente estará na faixa entre $$-126$$ e $$127$$ com um _bias_ de $$127$$ o que permitirá o uso de algorítmos entre $$1$$ e $$254$$, os valores $$0$$ e $$255$$ são reservados para representações especiais.
2. em precisão dupla um expoente estará na faixa entre $$-1022$$ e $$1023$$ com um _bias_ de $$1023$$ o que permitirá o uso de valores entre $$1$$ e $$2046$$, os valores $$0$$ e $$2047$$ são reservados para representações especiais.

Para fraseando um dos personagens do filme [Bolt](<https://pt.wikipedia.org/wiki/Bolt_(2008)>), a leitora deve colocar um _pin_ na frase: **são reservados para representações especiais** nós vamos voltar a isso mais trade. Por enquanto vamos voltar ao $$0,1_{10}$$. Este é valor numérico que mais irrita todo mundo que estuda este tópico. Deveria ser simples é acaba sendo muito complexo.

## De decimal para IEEE 754 na unha

A leitora terá que me dar um desconto, vou fazer em precisão simples. Haja zeros! E, por favor, preste atenção só vou fazer uma vez. :)

Antes de qualquer relação com a norma IEEE 754, vamos converter $$0,1_{10}$$ para binário. Começamos pela parte inteira deste número. Para isso vamos dividir o número inteiro repetidamente por dois, armazenar cada resto e parar quando o resultado da divisão, o quociente, for igual a zero e usar todos os restos para representar o número binário:

$$0 \div 2 = 0 + 0 \therefore 0_{10} = 0_2$$

Esta parte foi fácil $$0_{10}$$ é igual a $$0_2$$.

Em seguida precisamos converter a parte fracionária do número $$0,1$$ multiplicando este algoritmo repetidamente por dois até que a parte fracionária, aquilo que fica depois da vírgula, seja igual a zero e já vamos separando a parte inteira, resultado da multiplicação da parte fracionária. Vamos armazenar a parte inteira enquanto estamos multiplicando por dois a parte fracionária do resultado de cada operação anterior. Ou seja, começamos com $$0,1 \times 2 = 0,2$$ temos $$0$$ parte inteira do resultado da multiplicação e $$0,2$$ parte fracionária do resultado que vamos representar por $$0,1 \times 2 = 0 + 0,2$$ e assim sucessivamente:

<table class="table table-striped">
  <tr>
    <td>1. </td><td>$$0,1 × 2 = 0 + 0,2$$ </td>
    <td style="text-align: right;">13. </td><td>$$0,2 × 2 = 0 + 0,4$$ </td>
  </tr>
  <tr>
    <td>2. </td><td>$$0,2 × 2 = 0 + 0,4$$ </td>
    <td style="text-align: right;">14. </td><td>$$0,4 × 2 = 0 + 0,8$$ </td>
  </tr>
  <tr>
    <td>3. </td><td>$$0,4 × 2 = 0 + 0,8$$ </td>
    <td style="text-align: right;">15. </td><td>$$0,8 × 2 = 1 + 0,6$$ </td>
  </tr>
  <tr> 
    <td>4. </td><td>$$0,8 × 2 = 1 + 0,6$$ </td>
    <td style="text-align: right;">16. </td><td>$$0,6 × 2 = 1 + 0,2$$ </td>
  </tr>
  <tr>
    <td>5. </td><td>$$0,6 × 2 = 1 + 0,2$$ </td>
    <td style="text-align: right;">17. </td><td>$$0,4 × 2 = 0 + 0,8$$ </td>
  </tr>
  <tr>
    <td>6. </td><td>$$0,4 × 2 = 0 + 0,8$$ </td>
    <td style="text-align: right;">18. </td><td>$$0,8 × 2 = 1 + 0,6$$ </td>
  </tr>
  <tr> 
    <td>7. </td><td>$$0,8 × 2 = 1 + 0,6$$ </td>
    <td style="text-align: right;">19. </td><td>$$0,6 × 2 = 1 + 0,2$$ </td>
  </tr>
  <tr> 
    <td>8. </td><td>$$0,6 × 2 = 1 + 0,2$$ </td>
    <td style="text-align: right;">20. </td><td>$$0,2 × 2 = 0 + 0,4$$ </td>
  </tr>
  <tr>
    <td>9. </td><td>$$0,2 × 2 = 0 + 0,4$$ </td>
    <td style="text-align: right;">21. </td><td>$$0,4 × 2 = 0 + 0,8$$ </td>
  </tr>
  <tr>
    <td>10. </td><td>$$0,4 × 2 = 0 + 0,8$$ </td>
    <td style="text-align: right;">22. </td><td>$$0,8 × 2 = 1 + 0,6$$ </td>
  </tr>
  <tr>
    <td>11. </td><td>$$0,8 × 2 = 1 + 0,6$$ </td>
    <td style="text-align: right;">23. </td><td>$$0,6 × 2 = 1 + 0,2$$ </td>
  </tr>
  <tr>
    <td>12. </td><td>$$0,6 × 2 = 1 + 0,2$$ </td>
    <td style="text-align: right;">24. </td><td>$$0,2 × 2 = 0 + 0,4$$ </td>
  </tr>
  <tr>
    <td><td> </td> </td>
    <td style="text-align: right;">25. </td><td>$$0,4 × 2 = 0 + 0,8$$ </td>
  </tr>
</table>

Podemos continuar e não vamos conseguir encontrar um resultado de multiplicação cuja parte fracionária seja igual a $$0$$, contudo como na mantissa, em precisão simples, cabem 23 bits, acho que já chegamos a um nível suficiente de precisão. Precisamos agora ordenar todas as partes inteiras que encontramos para formar nosso binário:

$$0,1_{10} = 0,000110011001100110011001100_2$$

Resta normalizar este número. A leitora deve lembrar que a representação normal, não permite o $$0$$ como algarismo inteiro (antes da vírgula). O primeiro $$1$$ encontra-se na quarta posição logo:

$$0,0001 1001 1001 1001 1001 1001 100_2 = 1.1001 1001 1001 1001 1001 100_2 \times 2^{-4}$$

Precisamos agora normalizar nosso expoente. Como estamos trabalhando com precisão simples usaremos $$127_{10}$$ como _bias_. Como temos $$-4$$ teremos $$(-4+127)_{10} = 123_{10}$$ que precisa ser convertido para binário. Logo nosso expoente será $$01111011_2$$.

Até agora temos o sinal do número, $$0$$ e o expoente $$01111011$$ resta-nos terminar de trabalhar a mantissa. Podemos remover a parte inteira já que em binário esta será sempre $$1$$ devido ao $$0$$ não ser permitido. Feito isso, precisamos ajustar seu comprimento para $$23$$ bits e, temos nossa mantissa: $$10011001100110011001100$$. Linda! E resumo temos:

<table class="table table-striped"> 
  <tr>
    <th style="text-align:center !important;"> Elemento </th>
    <th style="text-align:center !important;"> Valor </th>
  </tr>
  <tbody>
    <tr><td style="text-align:center !important;">Sinal</td><td>$$(+) = 1$$</td></tr>
    <tr><td style="text-align:center !important;">Expoente</td><td>$$(123_{10}) = 01111011_2$$</td></tr>
    <tr><td style="text-align:center !important;">Mantissa</td><td>$$10011001100110011001100$$</td></tr>
    <tr><td style="text-align:right !important;">Total</td><td>$$32 \space bits$$</td></tr>
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
  <td> $$0$$ </td>
  <td> $$0$$ </td>
  <td> $$0$$ </td>
  <td> $$0$$ </td>
  <td> $$\pm 0$$ </td>
</tr>
<tr>
  <td> $$0$$ </td>
  <td> $$ \neq 0$$ </td>
  <td> $$0$$ </td>
  <td> $$ \neq 0$$ </td>
  <td> $$\pm \space Número \space Subnormal$$</td>
</tr>
<tr>
  <td> $$1-254$$ </td>
  <td> $$Qualquer \space valor$$ </td>
  <td> $$1-2046$$ </td>
  <td> $$Qualquer \space valor$$ </td>
  <td> $$\pm \space Número \space Normal $$</td>
</tr>
<tr>
  <td> $$255$$ </td>
  <td> $$0$$ </td>
  <td> $$2047$$ </td>
  <td> $$0$$ </td>
  <td> $$\pm \space Infinito$$</td>
</tr>
<tr>
<td> $$255$$ </td>
<td> $$\neq 0$$ </td>
<td> $$2047$$ </td>
<td> $$\neq 0$$ </td>
<td> $$NaN \space (Not \space a \space Number)$$</td>
</tr>
</tbody></table>

Resta-nos entender o que estes valores representam e seu impacto na computação.

### Números subnormais

Para a IEEE 754 normal é tudo que vimos anteriormente, todos os valores que podem ser representados usando as regras de sinal, expoente e mantissa de forma normalizada que a amável leitora teve a paciência de estudar junto comigo. Subnormal, ou não normalizado, é o termo que empregamos para indicar valores nos quais o campo expoente é preenchido com zeros. Se seguirmos a regra, para representar o algarismo $$0$$ o expoente deveria ser o $$-127$$. Contudo, para este caso, onde todo o campo expoente é preenchido com $$00000000_2$$ o expoente será $$-126$$. Neste caso especial, a mantissa não terá que seguir a obrigatoriedade de ter sempre o número $$1$$ como parte inteira. Não estamos falando de valores normalizados então o primeiro bit pode ser $$0_2$$ ou $$1_2$$. Estes números foram especialmente criados para aumentar a precisão na representação de números que estão no intervalo entre $$0_{10}$$ e $$1_{10}$$ melhorando a representação do conjunto dos números reais nesta faixa.

A leitora há de me perdoar novamente, a expressão subnormal é típica da norma IEEE 854 e não da IEEE 754, mas tomei a liberdade de usar esta expressão aqui por causa da melhor tradução.

### Zero

Observe que a definição de zero na norma IEEE 754 usa apenas o expoente e a mantissa e não altera nada no bit que é utilizado para indicar o sinal de um número. A consequência disto é que temos dois números binários diferentes um para $$+0$$ e outro para $$-0$$. A leitora deve pensar no zero como sendo apenas outro número subnormal que, neste caso acontece quando o expoente é $$0_2$$ e a mantissa é $$0_2$$. Sinalizar o zero não faz sentido matematicamente e tanto o $$+0$$ quanto o $$-0$$ representam o mesmo valor. Por outro lado, faz muita diferença do ponto de vista computacional e é preciso atenção para entender estas diferenças.

### Infinito

Outro caso especial do campo de exponentes é representado pelo valor $$11111111_2$$. Se o expoente for composto de $$8$$ algarismos $$1_2$$ e a mantissa for totalmente preenchida como $$0_2$$, então o valor representado será o infinito. Acompanhando o zero, o infinito pode ser negativo, ou positivo.

Neste caso, faz sentido matematicamente. Ou quase faz sentido. Não, não faz sentido nenhum! Não espere, faz sim! Droga infinito é complicado. A verdade é que ainda existem muitas controvérsias sobre os conceitos de infinito, mesmo os matemáticos não tem consenso sobre isso, a norma IEEE 754 com o $$\pm Infinito$$ atende ao entendimento médio do que representa o infinito.

Considerando que a linguagem de programação que você usa está de acordo com a norma IEEE 754, se você calcular o inverso de $$-0$$ deverá obter $$-Infinito$$, se calcular o inverso de $$+0$$ deve obter $$+Infinito$$. Por exemplo, em C++ teremos:

<iframe height="800px" width="100%" src="https://repl.it/@frankalcantara/infinityTeste?lite=true" scrolling="no" frameborder="no" allowtransparency="true" allowfullscreen="true" sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"></iframe>

### NaN (Not a Number)

O conceito de **NaN** foi criado para representar valores, principalmente resultados, que não correspondem a um dos números reais que podem ser representados em binário segundo a norma IEEE 754. Neste caso o expoente será completamente preenchido como $$1_2$$ e a mantissa será preenchida com qualquer valor desde que este valor não seja composto de todos os algarismos com o valor $$0_2$$. O bit relativo ao sinal não causa efeito no NaN. No entanto, existem duas categorias de NaN: QNaN _(Quiet NaN)_ e SNaN _(Signalling NaN)_.

O primeiro caso QNaN, _(Quiet NaN)_, ocorre quando o bit mais significativo da mantissa é $$1_2$$. O QNaN se propaga na maior parte das operações aritméticas e é utilizado para indicar que o resultado de uma determinada operação não é matematicamente definido. já o SNaN, _(Signalling NaN)_, que ocorre quando o bit mais significativo da mantissa é $$0_2$$ é utilizado para sinalizar alguma exceção como o uso de variáveis não inicializadas. Podemos sintetizar estes conceitos memorizando que QNaN indica operações indeterminadas enquanto SNaN indica operações inválidas.

  <table class="table table-striped">
        <tr style="text-align: center;">
          <th>Operação</th>
          <th>Resultado</th>
        </tr>
    <tbody>
        <tr>
          <td>$$(Número) \div (\pm Infinito)$$ </td>
          <td> $$0$$ </td>
        </tr>
        <tr>
            <td>$$(\pm Infinito) \times (\pm Infinito)$$</td>
            <td>$$\pm Infinito$$</td>
        </tr>
        <tr>
          <td>$$(\pm \neq 0) \div (\pm 0)$$</td>
          <td>$$\pm Infinito$$</td>
        </tr>
        <tr>
            <td>$$(\pm Número) \times (\pm Infinito)$$</td>
            <td>$$\pm Infinito$$</td>
        </tr>
        <tr>
          <td>$$(Infinito) + (Infinito)$$</td>
          <td>$$+Infinito$$</td>
        </tr>
        <tr>
          <td>$$(Infinito) - (-Infinito)$$</td>
          <td>$$+Infinito$$</td>
        </tr>
        <tr>
          <td>$$(-Infinito) + (-Infinito)$$</td>
          <td>$$-Infinito$$</td>
        </tr>
        <tr>
          <td>$$(-Infinito) - (Infinito)$$</td>
          <td>$$-Infinito$$</td>
        </tr>
        <tr>
          <td>$$(Infinito) - (Infinito)$$</td>
          <td>$$NaN$$</td>
        </tr>
        <tr>
          <td>$$(-Infinito) + (Infinito)$$</td>
          <td>$$NaN$$</td>
        </tr>
        <tr>
          <td>$$(\pm 0) \div (\pm 0)$$</td>
          <td>$$NaN$$</td>
        </tr>
        <tr>
           <td>$$(\pm Infinito) \div (\pm Infinito)$$</td>
           <td>$$NaN$$</td>
        </tr>
        <tr>
          <td>$$(\pm Infinito) \times (0)$$</td>
          <td>$$NaN$$</td>
        </tr>
        <tr>
            <td>$$(NaN) == (NaN)$$</td>
            <td>$$false$$</td>
        </tr>
    </tbody>
  </table>

Antes de chamar os aldeões e começar as fogueiras a amável leitora precisa levar em consideração as intensões que suportam a norma IEEE 754. Originalmente o objetivo era criar um ambiente padrão para a troca de números em ponto flutuante entre máquinas e softwares. Resolvendo milhares de problemas de compatibilidade que impediam o progresso da computação. E só. Neste interim, foi criado todo um padrão que permitisse operar com estes números em um grau de precisão aceitável para a imensa maioria das operações computacionais. Durante a criação da norma, ninguém se preocupou muito que valores como $$\pm Infinito$$ ou $$NaN$$ fossem ser usados para qualquer coisa diferente de criar interrupções e sinalizar erros. Foi o tempo que apresentou situações interessantes quando passamos a exigir dos nossos programas comportamentos numericamente corretos para a resolução de problemas complexos. A observação e o uso levou a percepção de alguns problemas intrigantes.

O $$-0$$ e o $$+0$$ representam exatamente o mesmo valor mas são diferentes $$-0 \neq +0$$ o que implica que em alguns casos, nos quais, mesmo que $$x=y$$ eventualmente podemos ter que $$\frac{1}{x} \neq \frac{1}{y}$$ para isso basta que algum momento durante o processo de computação $$x=-0$$ e $$y=+0$$ o que já é suficiente para criar uma grande quantidade de problemas. Antes de achar que isso é muito difícil lembre-se, por favor, que existe um numero próximo do infinito, só para ficar no dialeto que estamos usando, de funções que cruzam os eixos de um plano cartesiano. Um ponto antes estas funções estarão em $$-0$$ e um ponto depois em $$+0$$. Se tratarmos a existência do $$\pm 0$$ como interrupção ou alerta, podemos gerir estas ocorrências eficientemente e manter a integridade da matemática em nossos programas. Se formos observar cuidadosamente os cálculo e utilizar estes dois valores de zero de forma diferente então, teremos que prestar muita atenção nas equações que usaremos.

O infinito é outro problema. Pobres de nós! Estes conceitos foram inseridos na norma para permitir a concordância com a ideia que o infinito é uma quantidade, maior que qualquer quantidade possivelmente representada e atende a Teoria Axiomática de [Zermelo–Fraenkel](https://en.wikipedia.org/wiki/Zermelo%E2%80%93Fraenkel_set_theory). Isto é importante porque hoje, esta é a teoria axiomática da teoria dos conjuntos que suporta toda a matemática. Vamos deixar Zermelo–Fraenkel para um outro artigo já que este conhecimento não faz parte do cabedal de conhecimentos do programador mediano. Basta lembrar que as operações aritméticas são coerentes e que, na maior parte das linguagens é possível trabalhar isso como um alerta.

Por fim, temos o $$NaN$$ este valor indica uma operação inválida, como $$0 \div 0$$ ou $$\sqrt(-1)$$. Este valor será propagado ao longo da computação, assim que surgir como resultado, permitindo que a maioria das operações que resultem em $$NaN$$, ou usem este valor como operador, disparem algum tipo de interrupção, ou alerta, que indique que estamos trabalhando fora dos limites da matemática e, muitas vezes, da lógica. Novamente, os problemas ocorrem graças as decisões que tomamos quando criamos uma linguagem de programação. Hoje não é raro encontrar programas onde o valor $$NaN$$ seja utilizado como um valor qualquer inclusive em operações de comparação. Pobres de nós!

> Esta aritmética foi criada para que qualquer programador, mesmo o mais ignorante, fosse avisado de que algo estava fora do normal e não para que os meandros da teoria dos números fossem explorados. [William Kahan](https://amturing.acm.org/award_winners/kahan_1023746.cfm).

A leitora deve fazer um esforço para me compreender nesta última citação. [Não é uma citação literal](https://people.eecs.berkeley.edu/~wkahan/ieee754status/why-ieee.pdf), trata-se de uma paráfrase de um dos criadores da norma IEEE 754. Entendendo a intensão que suporta o ato, entendemos as consequências deste ato. A norma permite o uso de valores de forma algebricamente correta. E isto deveria bastar. Até que a gente encontra linguagens como o javascript.

```javascript
> typeof NaN
> "number"

> NaN = NaN
> false;
```

As duas operações estão perfeitamente corretas segundo a norma, mas não fazem nenhum sentido, pelo menos não para quem ignora a norma. Sim, realmente $$NaN$$ é um número e sim, $$NaN = NaN$$ é falso. Em [Javascript: the weird parts](https://charlieharvey.org.uk/page/javascript_the_weird_parts) Charlie Harvey explora muitas das incongruências encontradas no javascript apenas porque os interpretadores seguem rigidamente as normas sem atentar para as razões da existência destas normas.

Aqui eu usei exemplos do Python e do Javascript porque são mais fáceis de testar. Nenhuma linguagem de programação imperativa está livre destes problemas. Se quiser dar uma olhada em C++, no Windows, John D. Cook em [IEEE floating-point exceptions in C++](https://www.johndcook.com/blog/IEEE_exceptions_in_cpp/) mostra como fazer isso.

> Uma coisa deve ficar para sempre: não use pontos flutuantes para dinheiro e nunca use _float_ se o _double_ estiver disponível. Só use float se estiver escrevendo programas em ambientes muito, muito, muito limitados em memória.

Certa vez [Joel Spolsky](https://www.joelonsoftware.com/) criou o termo _leaky abstraction_ que eu aqui, em tradução livre vou chamar de **abstração fraca**. A computação é toda baseada em metáforas e abstrações. Uma abstração forte é aquela em que você usa uma ferramenta sem nunca ter que abrir e ver o que há lá dentro. Uma abstração fraca é aquela em que você tem que abrir a ferramenta antes de usar. **Pontos flutuantes são abstrações fracas**. E, apesar de todas as linguagens de programação que eu conheço usarem esta norma, a leitora não está obrigada a usar esta norma nos seus programas, mas isto é assunto para outro artigo.

# Referências

[Binary System](https://binary-system.base-conversion.ro/real-number-converted-from-decimal-system-to-32bit-single-precision-IEEE754-binary-floating-point.php?decimal_number_base_ten=0.2)

[Floating Point Numbers](https://floating-point-gui.de/formats/fp/)

[Floating Point](https://users.cs.duke.edu/~raw/cps104/TWFNotes/floating.html)

[What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)

[Floating Point Arithmetic: Issues and Limitations](https://docs.python.org/3/tutorial/floatingpoint.html)

[IEEE floating-point exceptions in C++](https://www.johndcook.com/blog/IEEE_exceptions_in_cpp/)

[Javascript: the weird parts](https://charlieharvey.org.uk/page/javascript_the_weird_parts)

[IEEE 754](https://en.wikipedia.org/wiki/IEEE_754#Rounding_rules)

[Taming Floating Point Error](https://www.johnbcoughlin.com/posts/floating-point-axiom/)

[Why do we need a floating-point arithmetic standard?](https://people.eecs.berkeley.edu/~wkahan/ieee754status/why-ieee.pdf)

[Zermelo–Fraenkel](https://en.wikipedia.org/wiki/Zermelo%E2%80%93Fraenkel_set_theory)
