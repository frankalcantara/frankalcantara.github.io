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

A memória disponível é limitada. Pelo menos ainda é limitada. Graças a isso, em algum momento temos que arredondar os números reais que queremos armazenar. A dificuldade está em decidir quanta informação precisamos armazenar e a forma como podemos armazenar esta informação para garantir que os computadores possam ser usados para resolver problemas reais.

<span>Foto de <a href="https://unsplash.com/@calliestorystreet?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Callie Morgan</a> on <a href="https://unsplash.com/s/photos/floating?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

Este problema de armazenamento não é exclusivo dos computadores. O caderno que a leitora usou para aprender a somar era limitado em quantidades de linhas e quantidade de páginas. Então vamos começar observando os números decimais (base $$10$$). Os números reais que nos interessam são aqueles que possuem uma parte inteira, antes da vírgula, e uma parte fracionária, depois da vírgula. Este termo fracionária tem origem na possibilidade de representar números reais na forma de operações com frações, formalmente, operações racionais. Assim temos:

$$0,125 = \frac{1}{10}+\frac{2}{100}+\frac{5}{1000} = \frac{1}{10^1}+\frac{2}{10^2}+\frac{5}{10^3}$$

> "Deus criou os inteiros, todo o resto é trabalho dos homens." Leopold Kronecker

E infelizmente o homem não trabalha muito bem. Os números reais, raramente podem ser exatamente representados por uma operação entre frações. Ou vice-versa.

Tome, por exemplo, a razão $$\frac{1}{6}$$ e tente representá-la em números reais sem arredondar, ou truncar. Verá, muito rapidamente que isto não é possível e que em algum ponto teremos que arredondar, ou truncar, para obtermos algo como $$0,166667$$. O momento onde vamos parar de dividir e arredondar, ou truncar, determina a precisão que usaremos para representar este número e a precisão será, por sua vez, definida pela aplicação, como usaremos, este número.

Em uma estrada, a diferença de um centímetro que existe entre $$12,00m$$ e $$12,01m$$ provavelmente não fará qualquer diferença. Se estivermos construíndo um motor, por outro lado, um erro de $$1cm$$ é a diferença entre funcionar ou explodir. Maximize este conceito imaginando-se no lugar de um um físico precisando utilizar a constante gravitacional. Neste caso, a leitora terá o desprazer de fazer contas com números como $0.00000000006667$.

Nossos computadores são binários, trabalham só e somente só com números na base $$2$$. Além disso, estas máquinas são limitadas pelo hardware e pelo software que executam. Ainda que exista um número infinito de números reais, representados por um número infinito de precisões diferentes, todo este universo numérico deverá caber em um espaço restrito e definido pela aritmética binária.

Os números na base dois podem ser representados por uma parte inteira e uma parte fracionária exatamente como fazemos com os números na base $$10$$. Dessa forma, o número $$0.001$$ na base $$2$$, pode ser representado por uma operação de frações:

$$0,001 = \frac{0}{2}+\frac{0}{4}+\frac{1}{8} = \frac{0}{2^1}+\frac{0}{2^2}+\frac{1}{2^3}$$

Se convertemos para binário, pura e simplesmente, os números $$0,125_10$$ e $$0,001_2$$ representam exatamente a mesma quantidade. A maioria dos números binários racionais, não pode ser representada de forma exata por uma operação de frações. Por exemplo, a fração $$(\frac{1}{3})_{10}$$ que seria representada por $$(\frac{1}{10})_{2}$$ terá que, em algum ponto ser arredondada, ou truncada. Definir este ponto, e a forma de armazenamento destes números é, ainda hoje, uma necessidade com a qual temos que lidar.

O problema é encontrar uma forma de representar todo o universo de números reais, em base $$10$$, em um espaço limitado em base $$2$$. Se pensarmos em uma correspondência de um para um, todo e qualquer número real deve ser armazenado no espaço de dados definido por um e apenas um endereço de memória. E, me permita adiantar um pouco as coisas. Isso é impossível.

<img class="img-fluid" src="{{ site.baseurl }}/assets/images/churchill1.jpg" alt="mostra as distribuição de bits o padrão ieee 754">

## E lá vem o homem com suas imperfeições

A norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) descreve como representar números em binário com precisão simples, $$32$$ bits, dupla, $$64$$ bits, ou quádrupla $$128$$ bits. Entretanto, antes de entrarmos nos meandros da representação binária de ponto flutuante, talvez seja uma boa ideia lembar o ensino médio. Foi lá que eu aprendi a notação científica. Pensando bem, hoje eu seria mais feliz se lá no Colégio Visconde de Mauá, alguém tivesse chamado a notação científica de representação de ponto flutuante.

A verdade é que na base $$10$$ somos treinados a usar pontos flutuantes, no que chamamos de notação científica, ou de notação de engenharia. Há uma pequena diferença entre a notação científica e a notação de engenharia que eu vou ignorar neste texto. Na notação científica temos a parte significativa do número, chamada de mantissa, e um expoente aplicado a uma potência de $$10$$. Chamar de mantissa é muito mais chique e vamos nos ater a isso. Usando a base $$10$$ poderíamos ter:

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

Com o tempo, e o uso, acabamos aprendendo formas mais eficiêntes de usar esta notação. Uma boa prática no uso da notação científica é deixar tantos dígitos significativos quanto necessário para o cálculo específico que pretendemos realizar. Lembre-se que esta notação foi criada quando a limitação de espaço era definida apenas pela capacidade de cálculo do ser humano e pelo espaço disponível no papel. Escolhemos a quantidade de números significativos de acordo com a aplicação. O $$\pi$$ é um bom exemplo de limitação relativa a aplicação.

Normalmente, um engenheiro civíl, ou eletricista, usa o $$\pi$$ como $$3.1416$$. Assim mesmo, arredondando a última casa. Cinco dígitos significativos resolvem a maior parte dos problemas no mundo sólido, real e visível. Este valor poderia ser representado por $$31,416 \times 10^{-1}$$, por mais estranho que pareça estaria correto. Leciono eletromagnetismo uso $$\pi = 3.1415926$$ rotineiramente, igualmente arredondando a última casa, esta prática definiu meu modo de pensar tão profundamente que achei muito estranho, e um tanto desnecessário, ter que usar $$\pi = 3.14159265359$$, truncado, sem nenhum arredondamento, para resolver alguns problemas de cosmologia. Diga-se de passagem, nesta disciplina o único que achava isso estranho era eu. Depois, eu descobri que o $$\pi$$ com 12 dígitos significativos só é útil na primeira disciplina do curso. Não é raro trabalhar com 30 dígitos de precisão. Neste ponto duas regras são muito importantes:

1. Cada problema tem a sua precisão particular e específica;
2. Não esqueça a primeira regra.

É preciso ficar claro que usei algumas notações científicas muito pouco usuais para o valor de $$\pi$$. O fiz apenas para que a leitora visse o ponto flutuar. Para que todos possam ser entendidos precisamos usar uma norma para representar nossos números em notação cientifica. Uma norma interessante seria usar $$1$$, e somente $$1$$, algarismo antes da vírgula e definir os algarismos depois da virgula de acordo com a precisão necessária e, principalmente, nunca deixar penas um zero antes da vírgula. Nunca representar a parte inteira do número fracionário com o zero. Adotando esta norma, $$3.1416$$ poderia ser representado por $$3.1416 \times 10^0$$, e estaria perfeitamente normalizado, ou por $$31,416\times 10^{-1}$$ que seria uma representação matematicamente válida, mas não normalizada. **Não estarão normalizados todos os números cuja parte inteira for $$0$$**.

Passou pela minha cabeça agora: está claro que a nomenclatura _ponto flutuante_ é importada do inglês? Se fosse em bom português seria vírgula flutuante. A gente vai falando, ou escrevendo, estas coisas, e nem se dá conta que não faz sentido no idioma de Camões e Braga.

A base não faz nenhuma diferença na notação científica. Números binários podem ser representados nesta notação. A leitora pode, por exemplo usar o número $$43.625_{10}$$ que, convertido para binário seria $$101011,101$$ e representá-lo em notação científica como $$1,01011101 \times 2^5$$. Guarde este número, vamos precisar dele.

> "Idealmente, um computador deve ser capaz de resolver qualquer problema matemático com a precisão necessária para este problema específico, sem desperdiçar memória, ou recursos computacionais." Anônimo.

Por acaso a amável leitora Lembra que eu falei da relação se um para um entre um número real e a sua representação? A norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) padronizou a representação binária de números de ponto flutuante e resolveu todos os problemas de compatibilidade entre hardware, software e mesmo entre soluções diferentes que existiam originalmente. Esta não é a única forma, não é a melhor forma mas, é de longe, a mais utilizada. Com esta norma em mãos, sabemos como representar uma faixa significativa de números e podemos determinar exatamente a precisão máxima possível para cada número e, principalmente, conhecemos os problemas inerentes a esta representação.

Quase esqueci. Não, a norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) não permite a representação de todo e qualquer número real. Temos um número infinito de valores na base $$10$$ representados em um número finito de valores na base $$2$$. Este é o momento em que a amável leitora deve ser perguntar: o que poderia dar errado?

## E os binários entram na dança

Precisamos converter os números reais na base $$10$$ que usamos diariamente para base $$2$$ que os computadores usam, armazenar estes números, realizar cálculos com os números em binário e finalmente converter estes valores para base $$10$$ de forma que seja possível entender a informação resultante do processo computacional. É neste vai e volta que os limites da norma [IEEE 754](http://en.wikipedia.org/wiki/IEEE_754-2008) são testados e, não raramente, causam alguns espantos e muitos problemas.

Tomemos, por exemplo o número decimal $$0,1_{10}$$. Usando o [Decimal to Floating-Point Converter](https://www.exploringbinary.com/floating-point-converter/) para poupar tempo, e precisão dupla, já explico isso, podemos ver que:

$$0,1_{10} = (0.0001100110011001100110011001100110011001100110011001101)_2$$

Ou seja, $$0,1_{10}$$ será guardado em memória como $$(0.0001100110011001100110011001100110011001100110011001101)_2$$. Um belo de um número binário que, em algum momento será convertido para decimal resultado em:

$$(0.0001100110011001100110011001100110011001100110011001101)_2 = (0.1000000000000000055511151231257827021181583404541015625)_{10}$$

Eita! Virou outra coisa. Uma coisa bem diferente. Eis por que em Python, acabamos encontrando coisas como:

```python
>0.1 * 3
>0.30000000000000004
```

Se não acreditar em mim, tente você mesmo, direto na linha de comando do Python ou em alguma célula do [Google Colab](https://colab.research.google.com/).

> Talvez esta seja uma boa hora para se levantar, tomar um copo d'água e pensar sobre jornalismo, contabilidade, educação física, ou qualquer outra opção de carreira que não envolva computação tão diretamente. Vai lá! Eu espero. Tem pressa não!

Muitas linguagens de programação, o Python, inclusive, conhecem um conjunto de valores onde erros deste tipo ocorrem e arredondam, ou truncam, o resultado para que você veja o resultado correto. Ou ainda, simplesmente limitam o que é exposto para outras operações, como se estivessem limitando a precisão do cálculo ou dos valores armazenados. Esta foi uma opção pouco criativa adotada por muitos compiladores e interpretadores que acaba criando mais problemas que soluções. Para ver um exemplo, use a fração $$frac{1}{10}$$, ainda em Python e reproduza as seguintes operações:

```python
> 1 / 10
>0.1
```

Vamos experimentar alguma coisa um pouco mais complicada, ainda utilizando o Python:

```python
a = 1/10
print( a)
print ("{0:.20f}".format(a))
0.10000000000000000555
```

A diferença entre estes dois últimos exemplos está na saída formatada para forçar a exibição de mais casas decimais mostrando que o erro está lá. Você não está vendo este erro, o interpretador vai tentar não permitir que este erro se propague, mas ele está lá.

> "Os interpretadores e compiladores são desenvolvidos por seres humanos, tão confiáveis quanto pescadores e caçadores. Não acredite em histórias de pescaria nem em compiladores" Frank de Alcantara.

_Isto não é uma exclusividade do Python_, a grande maioria das linguagens de programação, sofre de problemas semelhantes em maior ou menor número. Graças as limitações de espaço e as soluções encontradas pela IEEE 754. Mesmo que os compiladores e interpretadores se esforcem para não permitir a propagação deste erro se você fizer uma operação com o valor $$0.10000000000000000555$$ com algum outro valor que exija toda esta precisão, o erro estará lá.

Volte um pouquinho e reveja o que aconteceu, no Python, quando operamos $$0.1 * 3$$. A leitora deve observar que, neste caso, os dois operandos estão limitados e são exatos. O erro ocorre por que a conversão de $$0.1_{10}$$ para binário não é exata e somos forçados a parar em algum ponto e, ou truncamos ou arredondamos o valor. Digamos que paramos em: $$0.0001100110011001101_2$$. Se fizermos isso e convertemos novamente para o decimal o $$0.1$$ será convertido em $$0.1000003814697265625$$. E lá temos um baita de um erro. Se a conversão for feita usando os padrões impostos pela IEEE 754 os valores ficam um pouco diferentes, o valor $$0.1$$ será armazenado em um sistema usando a IEEE 754 como:

1. em precisão simples: $$00111101 11001100 11001100 11001101_2$$;
2. em precisão dupla: $$00111111 10111001 10011001 10011001 10011001 10011001 10011001 10011010_2$$.

Que quando convertidos novamente para binário representarão $$0.100000001490116119385$$ isso implica em um erro 256 vezes menor que o erro que obtemos com a conversão direta e os poucos bits que usamos na conversão manual. Nada mal! Vamos ver se entendemos como esta conversão pode ser realizada usando o $$0,1$$. Mas antes divirta-se um pouco vendo o resultado graças a IEEE 754 para: $$0,2$$; $$0,4$$ e $$0,8$$ usando o excelente [Float Point Exposed](https://float.exposed). Como disse antes: tem pressa não!

## Entendendo a IEEE 754

A norma IEEE 754 especifica 5 formatos binários: meia precisão - $$16$$ bits; precisão simples - $$32$$ bits; precisão dupla - $$64$$ bits; precisão quadrupla - $$128$$ bits e precisão óctupla - $$256$$ bits. A leitora há de me perdoar mas vamos nos limitar as duas estruturas de bits mais comuns da IEEE 754, à saber:

<img class="img-fluid" src="{{ site.baseurl }}/assets/images/ieee754.png" alt="mostra as distribuição de bits o padrão ieee 754">

Um valor real, na base $$10$$ será convertido em binário e ocupará o espaço de $$32$$, ou $$64$$ bits dependendo da precisão escolhida e das capacidades físicas da máquina que irá armazenar este dado. Nos dois casos, o primeiro bit, o mais significativo será reservado para indicar o sinal do número armazenado. Desta forma, quando encontramos o $1$ neste bit temos a representação de um valor negativo armazenado. Os próximos $$8$$ bits, para a precisão simples ou $$11$$ bits para a precisão dupla, são reservados para o expoente que usaremos para a representação em ponto flutuante. Volto ao expoente já, já. Agora vamos dar uma olhada na mantissa, a parte significativa do valor que estamos utilizando.

A terceira seção, que comporta $$23$$ bits em precisão simples e $$52$$ em precisão dupla é chamada de mantissa e contém o binário equivalente aos algoritmos significativos do nosso número. A leitora deve ser lembrar que eu pedi para guardar o número $$1,01011101 \times 2^5$$, já que a nossa mantissa, em precisão simples tem espaço para $$23$$ bits poderíamos, simplesmente, armazenar $$10101110100000000000000$$. E, neste ponto, temos que parar e pensar um pouco.

Na notação científica não podemos ter um zero antes de vírgula. Nós definimos isso anteriormente. O mesmo deve ser considerado para a notação científica em binário. Com uma grande diferença. Se o algarismo antes da vírgula não pode ser um zero ele obrigatóriamente deve ser o $$1$$. Ou seja, **a mantissa não precisa armazenar o algoritmo antes da vírgula** nunca. Sendo assim, para armazenar a mantissa de $$1,01011101 \times 2^5$$ vamos utilizar apenas $$01011101$$ que resultará em $$101011101000000000000000$$ uma precisão maior graças ao zero a mais. A leitora tinha contado os zeros? Está claro que preenchemos os $$32$$ bits do mais significativo para o menos significativo por que estamos colocando algoritmos depois da vírgula?

A mantissa é simples, a leitora, se estiver curiosa, pode ler sobre a relação entre casas decimais em binário e as casas decimais na base dois [neste link](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470124604.app15). Posso apenas adiantar que esta relação tende a $$log_2(10) \equiv 3.32$$. Isto implica na necessidade de aproximadamente $$3.32$$ vezes mais algoritmos em binário que em decimal para representar a mesma precisão. Até aqui foi tranquilo, a leitora deve ser preparar para os expoentes. Só para lembrar, temos $$8$$ bits em precisão simples e $$11$$ bit em precisão dupla.

**Considerando a precisão simples**, entre os $$8$$ bits reservados para a representação do expoente não existe um bit que seja específico para indicar expoentes negativos. Em vez disso, os valores são representados neste espaço de $$8$$ bits em uma notação chamada de **excess-127**. Nesta notação, utilizamos um número inteiro de $$8$$ bits cujo valor sem sinal é representado por $$M-127$$. O valor $$01111111_2$$ equivalente ao valor $$127_{10}$$ representa o expoente $$0_{10}$$, o valor $$01000000_2$$ equivalente a $$128_{10}$$, representa o expoente $$1_{10}$$, enquanto o valor $$01111110_2$$ equivalente a $$126_{10}$$ representa o expoente $$-1_{10}$$ e por ai vamos. Em outras palavras, para representar o expoente $$0$$ armazenamos nos bits reservados para isso o valor binário $$M=01111111_2$$ equivalente ao $$127_{10}$$ e o expoente será dado por $$M$$ subtraído do valor $$127_{10}$$, ou seja $$0$$. Usando esta técnica **excess-127 ou bias** teremos uma faixa de expoentes que variam $$2^{-126}$$ e $$2^{128}$$ para a precisão simples. Parece complicado e é mesmo.

**No caso da precisão dupla** o raciocínio é exatamente o mesmo exceto que o espaço é de $$11$$ bits e o _bias_ é de 1023 (excess-1023). Com $$11$$ bits conseguimos representar valores entre $$0$$ e $$2047$$. Neste caso, o $$M=1023$$ irá representar o valor $$0$$. Com a precisão dupla poderemos representar expoentes entre $$-1022$$ e $$1023$$. Em resumo:

1. em precisão simples um expoente estará na faixa entre $$-126$$ e $$127$$ com um _bias_ de $$127$$ o que permitirá o uso de algorítmos entre $$1$$ e $$254$$, os valores $$0$$ e $$255$$ são reservados para representações especiais.
2. em precisão dupla um expoente estará na faixa entre $$-1022$$ e $$1023$$ com um _bias_ de $$127$$ o que permitirá o uso de algorítmos entre $$1$$ e $$2046$$, os valores $$0$$ e $$2047$$ são reservados para representações especiais.

Para fraseando um dos personagens do filme Bolt, a leitora deve colocar um _pin_ na frase: **são reservados para representações especiais** nós vamos voltar a isso mais trade. Por enquanto vamos voltar ao $$0,1_{10}$$. Este é valor o que mais irrita todo mundo que estuda este tópico.

## De decimal para IEEE na unha

A leitora terá que me dar um desconto, vou fazer em precisão simples. Haja zeros!

Antes de qualquer relação com a norma IEEE 754, primeiro vamos converter $$0,1_{10}$$ para binário. Começamos pela parte inteira deste número. Para isso vamos dividir o número inteiro repetidamente por dois, armazenar cada resto e parar quando o resultado da divisão, o quociente for igual a zero e usar todos os restos para representar o número binário:

$$0 \div 2 = 0 + 0 \therefore 0_{10} = 0_2$$

Em seguida precisamos converter a parte fracionária do número $$0,1$$ multiplicando este algoritmo repetidamente por dois até que a parte fracionária seja igual a zero e já vamos separando a parte inteira, resultado da multiplicação da parte fracionária, a parte inteira vamos armazenar enquanto estaremos multiplicando por dois a parte fracionária do resultado de cada operação anterior. Ou seja, começamos com $$0,1 \times 2 = 0,2$$ temos $$0$$ parte inteira do resultado da multiplicação e $$0,2$$ parte fracionária do resultado da multiplicação que vamos representar por $$0,1 \times 2 = 0 + 0,2$$ e sucessivamente:

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

Podemos continuar e não vamos conseguir encontrar um resultado de multiplicação cuja parte fracionária seja igual a $$0$$ contudo, como na mantissa, precisão simples, cabem 23 bits, acho que já chegamos a um nível suficiente de precisão. Precisamos agora ordenar todas as partes inteiras que encontramos para formar nosso binário:

$$0,1_{10} = 0,0001 1001 1001 1001 1001 1001 100_2$$

Resta normalizar este número. A leitora deve lembrar que a representação normal, não permite o $$0$$ como algarismo inteiro (antes da vírgula). o primeiro $$1$$ encontra-se na quarta posição logo:

$$0,0001 1001 1001 1001 1001 1001 100_2 = 1.1001 1001 1001 1001 1001 100_2 \times 2^-4$$

Precisamos agora normalizar nosso expoente. Como estamos trabalhando com precisão simples usaremos 127 como _bias_. Como temos $$-4$$ teremos $$(-4+127)_{10} = 123_{10}$$ que precisa ser convertido para binário. Logo nosso expoente será $$01111011$$.

Até agora temos o sinal do número, $$0$$ e o expoente $$01111011$$ resta-nos terminar de trabalhar a mantissa. Podemos remover a parte inteira já que em binário esta será sempre $$1$$ já que o $$0$$ não é permitido pela norma. Feito isso, precisamos ajustar seu comprimento para 23 bits e, temos nossa mantissa: $$10011001100110011001100$$ e teremos:

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

A leitora deve lembrar da expressão que pedi que colocasse um pin: **são reservados para representações especiais**. Está na hora de tocar neste assunto delicado. A verdade é que não utilizamos a IEEE 754 apenas para números, utilizamos para representar todos os valores possíveis de representação computacional. Isto quer dizer que temos que armazenar o zero, o infinito e valores que não são numéricos, os famosos **NAN**, abreviação da expressão em inglês que significa **não é um número** _(Not A Number)_. Estes valores especiais estão sintetizados na tabela a seguir:

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

### Números subnormais

Para a IEEE 754 normal é tudo que vimos anteriormente, todos os valores que podem ser representados usando as regras de sinal, expoente e mantissa que a amável leitora teve a paciência de estudar junto comigo. Subnormal, ou não normalizado, é o termo que empregamos para indicar valores nos quais o campo expoente é preenchido com zeros. Se seguirmos a regra, neste caso o expoente deveria ser o $$-127$$. Contudo, para este caso, o expoente será $$-126$$ e, neste caso especial, a mantissa não terá que seguir a obrigatoriedade de ter sempre o número $$1$$ como parte inteira. Estes números foram especialmente criados para aumentar a precisão na representação de números que estão no intervalo entre $$0_{10}$$ e $$1_{10}$$ aumentando a precisão nesta região do conjunto dos números reais. A leitora há de me perdoar novamente, a expressão subnormal é típica da norma IEEE 854 e não da IEEE 754, mas tomei a liberdade de usar esta expressão aqui por causa da melhor tradução.

### Zero

Observe que a definição de zero na norma IEEE 754 usa apenas o expoente e a mantissa e não altera nada no bit que é utilizado para indicar o sinal de um número. A consequência disto é que temos dois números binários diferentes um para $$+0$$ e outro para $$-0$$. A leitora deve pensar no zero como sendo apenas outro número subnormal que, neste caso acontece quando o expoente é $$0_2$$ e a mantissa é $$0_2$$. Sinalizar o zero não faz sentido matematicamente e tanto o $$+0$$ quanto o $$-0$$ representam o mesmo valor. Programaticamente existem diferenças entre estes valores e é preciso atenção para entender estas diferenças.

### Infinito

Outro caso especial do campo de exponentes é representado pelo valor $$11111111_2$$. Se o expoente for composto de $$8$$ algarismos $$1$$ e a mantissa for totalmente preenchida como $$0_2$$, então o valor representado será o infinito. Acompanhando o zero, o infinito pode ser negativo, ou positivo. Neste caso, faz sentido matematicamente. Ou quase faz sentido. Não não faz sentido nenhum! Não espere, faz sim! Droga infinito é complicado.

### NaN (Not a Number)

O conceito de NaN foi criado para representar valores que não correspondem a um dos números reais que podem ser representados em binário segundo a norma IEEE 754. Neste caso o expoente será completamente preenchido como $$1$$ e a mantissa será preenchida com qualquer valor desde que este valor não seja composto de todos os algarismos com o valor $$0_2$$. O bit relativo ao sinal não causa efeito no NaN. No entanto, existem duas categorias de NaN: QNaN _(Quiet NaN)_ e SNaN \_(Signalling NaN).

O primeiro caso QNaN _(Quiet NaN)_ ocorre quando o bit mais significativo da mantissa é $$1_2$$. O QNaN se propaga na maior parte das operações aritméticas e é utilizado para indicar que o resultado de uma determinada operação não é matematicamente definido. já o SNaN, que ocorre quando o bit mais significativo da mantissa é $$0_2$$ é utilizado para sinalizar alguma exceção como o uso de variáveis não inicializadas. Podemos sintetizar estes conceitos memorizando que QNaN indica operações indeterminadas enquanto SNaN indica operações inválidas.

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

Antes de chamar os aldeões e começar as fogueiras a amável leitora precisa levar em consideração as intensões que suportam a norma IEEE 754. Originalmente o objetivo era criar um ambiente padrão para a troca de números em ponto flutuante entre máquinas e softwares. Resolvendo milhares de problemas de compatibilidade que impediam o progresso da computação. Além disso, em nenhum momento alguém pensou, originalmente, que valores como $$\pm Infinito$$ ou $$NaN$$ fossem ser usados para qualquer coisa diferente de criar interrupções e sinalizar erros. Foi o tempo que apresentou situações interessantes.

O $$-0$$ e o $$+0$$ representam exatamente o mesmo valor mas são diferentes $$-0 \neq +0$$ o que implica que em alguns casos, nos quais, mesmo que $$x=y$$ eventualmente podemos ter que $$\frac{1}{x} \neq \frac{1}{y}$$ para isso basta que algum momento durante o processo de computação $$x=-0$$ e $$y=+0$$ o que já é suficiente para criar uma grande quantidade de problemas. Antes de achar que isso é muito difícil lembre-se, por favor, que existe um numero próximo do infinito, só para ficar no tema, de funções que cruzam os eixos um ponto antes estas funções estarão em $$-0$$ e um ponto depois em $$+0$$. Se tratarmos a existência do $$\pm 0$$ como interrupção ou alerta, podemos gerir estas ocorrências eficientemente e manter a integridade da matemática em nossos programas.

O infinito é outro problema. Nem os matemáticos tem consenso sobre isso. Pobre de nós! Estes conceitos foram inseridos na norma para permitir a concordância com a ideia que o infinito é uma quantidade, maior que qualquer quantidade possivelmente representada e atende a Teoria Axiomática de Zermelo–Fraenkel. Isto é importante por que hoje, esta é a teoria axiomática da teoria dos conjuntos que suporta toda a matemática. Vamos deixar Zermelo–Fraenkel para um outro artigo. Neste momento é importante lembrar que, ainda que existam coerências entre as operações realizadas com o valor $$\pm Infinito$$ não era o propósito da norma favorecer estas operações.

Por fim, temos o $$NaN$$ este valor indica uma operação inválida, como $$0 \div 0$$ ou $$\sqrt(-1)$$. Este valor será propagado ao longo da computação indicando que a maioria das operações que resultem em $$NaN$$, usem este valor como operador, disparem algum tipo de interrupção, ou alerta, que indique que estamos trabalhando fora dos limites da matemática e, muitas vezes, da lógica. Novamente, os problemas ocorrem graças as decisões que tomamos quando criamos uma linguagem de programação.

> Esta aritmética foi criada para que qualquer programador, mesmo o mais ignorante, fosse avisado de que algo estava fora do normal e não para que os meandros da teoria dos números fossem explorados. [William Kahan](https://amturing.acm.org/award_winners/kahan_1023746.cfm).

A leitora deve fazer um esforço para me compreender nesta última citação. [Não é uma citação literal](https://people.eecs.berkeley.edu/~wkahan/ieee754status/why-ieee.pdf), mas uma paráfrase de um dos criadores da norma IEEE 754. Entendendo a intensão que suporta o ato, entendemos as consequências. A norma permite o uso de valores de forma algebricamente correta. E isto deveria bastar. Até que a gente encontra linguagens como o javascript.

```javascript
> typeof NaN
> "number"

> NaN = NaN
> false;
```

As duas operações estão perfeitamente corretas segundo a norma, mas não fazem nenhum sentido, pelo menos não para quem ignora a norma. Sim, realmente $$NaN$$ é um número e sim, $$NaN = NaN$$ é falso. Em [Javascript: the weird parts](https://charlieharvey.org.uk/page/javascript_the_weird_parts) Charlie Harvey explora muitas das incongruências encontradas no javascript apenas por que os interpretadores seguem rigidamente as normas sem atentar para as razões da existência destas normas.

Aqui eu usei exemplos do Python e do Javascript apenas por que são mais fáceis de testar. Nenhuma linguagem de programação imperativa está livre destes problemas. Se quiser dar uma olhada em C++ no Windows John D. Cook em [IEEE floating-point exceptions in C++](https://www.johndcook.com/blog/IEEE_exceptions_in_cpp/) mostra como fazer isso.

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

[Why do we need a floating-point arithmetic standard?](https://people.eecs.berkeley.edu/~wkahan/ieee754status/why-ieee.pdf)
