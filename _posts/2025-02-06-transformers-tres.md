---
layout: post
title: Transformers - Mude o Título
author: frank
categories:
    - disciplina
    - Matemática
    - artigo
tags:
    - C++
    - Matemática
    - inteligência artificial
image: assets/images/trans1.webp
featured: false
rating: 0
description: ""
date: 2025-02-09T22:55:34.524Z
preview: ""
keywords: ""
toc: true
published: false
beforetoc: ""
lastmod: 2025-04-01T13:28:46.044Z
---

### Modelo de Sequência de Primeira Ordem

Podemos deixar as matrizes de lado por um minuto e voltar ao que realmente nos importa, sequências de palavras. Imagine que, à medida que começamos a desenvolver nossa interface de computador de linguagem natural, queremos lidar apenas com três comandos possíveis:

Mostre-me meus diretórios, por favor.
Mostre-me meus arquivos, por favor.
Mostre-me minhas fotos, por favor.
Nosso tamanho de vocabulário agora é sete:
{diretórios, arquivos, me, meu, fotos, por favor, mostrar}.
Uma maneira útil de representar sequências é com um modelo de transição. Para cada palavra no vocabulário, ele mostra qual é a próxima palavra provável. Se os usuários perguntam sobre fotos metade do tempo, arquivos 30% do tempo e diretórios o resto do tempo, o modelo de transição será assim. A soma das transições a partir de qualquer palavra sempre somará um.

Modelo de cadeia de Markov

Esse modelo de transição específico é chamado de cadeia de Markov, porque satisfaz a propriedade de Markov de que as probabilidades para a próxima palavra dependem apenas das palavras recentes. Mais especificamente, é um modelo de Markov de primeira ordem porque considera apenas a palavra mais recente. Se considerasse as duas palavras mais recentes, seria um modelo de Markov de segunda ordem.

Nossa pausa das matrizes acabou. Acontece que as cadeias de Markov podem ser expressas convenientemente na forma de matriz. Usando o mesmo esquema de indexação que usamos ao criar vetores one-hot, cada linha representa uma das palavras em nosso vocabulário. O mesmo acontece com cada coluna. A matriz do modelo de transição trata uma matriz como uma tabela de consulta. Encontre a linha que corresponde à palavra que te interessa. O valor em cada coluna mostra a probabilidade da próxima palavra. Como o valor de cada elemento na matriz representa uma probabilidade, eles sempre cairão entre zero e um. Como as probabilidades sempre somam um, os valores em cada linha sempre somarão um.

Matriz de transição

Na matriz de transição aqui, podemos ver a estrutura de nossas três frases claramente. Quase todas as probabilidades de transição são zero ou um. Existe apenas um lugar na cadeia de Markov onde ocorre ramificação. Depois de meu, as palavras diretórios, arquivos ou fotos podem aparecer, cada uma com uma probabilidade diferente. Além disso, não há incerteza sobre qual palavra virá a seguir. Essa certeza é refletida tendo quase todos uns e zeros na matriz de transição.

Podemos revisitar nosso truque de usar a multiplicação de matrizes com um vetor one-hot para extrair as probabilidades de transição associadas a qualquer palavra. Por exemplo, se quisermos apenas isolar as probabilidades de qual palavra vem depois de meu, podemos criar um vetor one-hot representando a palavra meu e multiplicá-lo por nossa matriz de transição. Isso extrai a linha relevante e mostra a distribuição de probabilidade do que será a próxima palavra.

Consulta de probabilidade de transição

Modelo de Sequência de Segunda Ordem
Prever a próxima palavra com base apenas na palavra atual é difícil. Isso é como prever o resto de uma música depois de receber apenas a primeira nota. Nossas chances são muito melhores se pudermos pelo menos obter duas notas para continuar.

Podemos ver como isso funciona em outro modelo de linguagem de brinquedo para nossos comandos de computador. Esperamos que este só veja duas frases, em uma proporção de 40/60.

Verifique se a bateria acabou, por favor.
Verifique se o programa foi executado, por favor.
Uma cadeia de Markov ilustra um modelo de primeira ordem para isso.

Outra cadeia de Markov de primeira ordem

Aqui podemos ver que, se nosso modelo olhasse para as duas palavras mais recentes, em vez de apenas uma, ele poderia fazer um trabalho melhor. Quando encontrar bateria executada, ele sabe que a próxima palavra será abaixo e, quando vê programa executado, a próxima palavra será por favor. Isso elimina uma das ramificações no modelo, reduzindo a incerteza e aumentando a confiança. Olhar para trás duas palavras transforma isso em um modelo de Markov de segunda ordem. Ele fornece mais contexto sobre o qual basear as previsões da próxima palavra. Os modelos de Markov de segunda ordem são mais desafiadores de desenhar, mas aqui estão as conexões que demonstram seu valor.

Cadeia de Markov de segunda ordem

Para destacar a diferença entre os dois, aqui está a matriz de transição de primeira ordem,

Outra matriz de transição de primeira ordem

e aqui está a matriz de transição de segunda ordem.

Matriz de transição de segunda ordem

Observe como a matriz de segunda ordem tem uma linha separada para cada combinação de palavras (a maioria das quais não está mostrada aqui). Isso significa que, se começarmos com um tamanho de vocabulário de N, a matriz de transição terá N^2 linhas.

O que isso nos oferece é mais confiança. Existem mais uns e menos frações no modelo de segunda ordem. Existe apenas uma linha com frações, uma ramificação em nosso modelo. Intuitivamente, olhar para duas palavras em vez de apenas uma fornece mais contexto, mais informações sobre as quais basear um palpite de próxima palavra.

Modelo de Sequência de Segunda Ordem com Saltos
Um modelo de segunda ordem funciona bem quando só precisamos olhar para trás duas palavras para decidir qual palavra vem a seguir. E quanto a quando precisamos olhar para trás mais? Imagine que estamos construindo outro modelo de linguagem. Este só precisa representar duas frases, cada uma com a mesma probabilidade de ocorrer.

Verifique o log do programa e descubra se ele foi executado, por favor.
Verifique o log da bateria e descubra se ele acabou, por favor.
Neste exemplo, para determinar qual palavra deve vir depois de executado, teríamos que olhar para trás 8 palavras no passado. Se quisermos melhorar nosso modelo de linguagem de segunda ordem, poderíamos, é claro, considerar modelos de terceira e ordens superiores. No entanto, com um tamanho de vocabulário significativo, isso exige uma combinação de criatividade e força bruta para executar. Uma implementação ingênua de um modelo de oitava ordem teria N^8 linhas, um número ridículo para qualquer vocabulário razoável.

Em vez disso, podemos fazer algo inteligente e criar um modelo de segunda ordem, mas considerar as combinações da palavra mais recente com cada uma das palavras que vieram antes. Ainda é de segunda ordem porque estamos considerando apenas duas palavras de cada vez, mas nos permite alcançar mais e capturar dependências de longo alcance. A diferença entre este modelo de segunda ordem com saltos e um modelo de ordem superior completo é que descartamos a maior parte das informações de ordem das palavras e combinações de palavras anteriores. O que resta ainda é bastante poderoso.

Cadeias de Markov falham completamente agora, mas ainda podemos representar o elo entre cada par de palavras anteriores e as palavras que se seguem. Aqui, dispensamos os pesos numéricos e, em vez disso, mostramos apenas as setas associadas aos pesos diferentes de zero. Pesos maiores são mostrados com linhas mais grossas.

Votação de recursos de segunda ordem com saltos

Aqui está como pode ser uma matriz de transição.

Matriz de transição de segunda ordem com saltos

Esta visão mostra apenas as linhas relevantes para prever a palavra que vem depois de executado. Ele mostra instâncias em que a palavra mais recente (executado) é precedida por cada uma das outras palavras no vocabulário. Apenas os valores relevantes são mostrados. Todas as células vazias são zeros.

A primeira coisa que se torna aparente é que, ao tentar prever a palavra que vem depois de executado, não olhamos mais apenas uma linha, mas sim um conjunto inteiro delas. Saímos do reino de Markov agora. Cada linha não representa mais o estado da sequência em um ponto particular. Em vez disso, cada linha representa uma das muitas características que podem descrever a sequência em um ponto particular. A combinação da palavra mais recente com cada uma das palavras que a precederam forma uma coleção de linhas aplicáveis, talvez uma grande coleção. Devido a essa mudança de significado, cada valor na matriz não representa mais uma probabilidade, mas sim um voto. Os votos serão somados e comparados para determinar as previsões da próxima palavra.

A próxima coisa que se torna aparente é que a maioria das características não importa. A maioria das palavras aparece em ambas as frases e, portanto, o fato de terem sido vistas não ajuda a prever o que vem a seguir. Elas têm todas um valor de 0,5. As únicas duas exceções são bateria e programa. Eles têm alguns pesos de 1 e 0 associados aos dois casos que estamos tentando distinguir. A característica bateria, executado indica que executado foi a palavra mais recente e que bateria ocorreu em algum lugar anterior na frase. Esta característica tem um peso de 1 associado a abaixo e um peso de 0 associado a por favor. Da mesma forma, a característica programa, executado tem o conjunto oposto de pesos. Esta estrutura mostra que é a presença dessas duas palavras anteriores na frase que é decisiva para prever qual palavra vem a seguir.

Para converter este conjunto de características de pares de palavras em uma estimativa da próxima palavra, os valores de todas as linhas relevantes precisam ser somados. Somando a coluna, a sequência Verifique o log do programa e descubra se ele foi executado gera somas de 0 para todas as palavras, exceto 4 para abaixo e 5 para por favor. A sequência Verifique o log da bateria e descubra se ele acabou faz o mesmo, exceto com 5 para abaixo e 4 para por favor. Ao escolher a palavra com o maior total de votos como a previsão da próxima palavra, este modelo nos dá a resposta correta, apesar de ter uma dependência profunda de oito palavras.

Mascaramento
Em consideração mais cuidadosa, isso é insatisfatório. A diferença entre um total de votos de 4 e 5 é relativamente pequena. Sugere que o modelo não está tão confiante quanto poderia estar. E em um modelo de linguagem maior e mais orgânico, é fácil imaginar que tal diferença pequena poderia se perder no ruído estatístico.

Podemos aguçar a previsão eliminando todos os votos de características não informativas. Com exceção de bateria, executado e programa, executado. É útil lembrar neste ponto que extraímos as linhas relevantes da matriz de transição multiplicando-a por um vetor que mostra quais características estão atualmente ativas. Para este exemplo até agora, usamos o vetor de características implícito mostrado aqui.

Vetor de seleção de características

Ele inclui um para cada característica que é uma combinação de executado com cada uma das palavras que o precedem. Qualquer palavra que venha depois dele não é incluída no conjunto de características. (No problema de previsão da próxima palavra, essas ainda não foram vistas e, portanto, não é justo usá-las para prever o que vem a seguir.) E isso não inclui todas as outras combinações possíveis de palavras. Podemos ignorar com segurança essas para este exemplo porque todas serão zero.

Para melhorar nossos resultados, também podemos forçar as características não úteis a zero criando uma máscara. É um vetor cheio de uns, exceto nas posições que você gostaria de ocultar ou mascarar, e essas são definidas como zero. Em nosso caso, gostaríamos de mascarar tudo, exceto bateria, executado e programa, executado, as únicas duas características que foram de alguma ajuda.

Atividades de características mascaradas

Para aplicar a máscara, multiplicamos os dois vetores elemento por elemento. Qualquer valor de atividade de característica em uma posição não mascarada será multiplicado por um e deixado inalterado. Qualquer valor de atividade de característica em uma posição mascarada será multiplicado por zero e, portanto, forçado a zero.

A máscara tem o efeito de ocultar grande parte da matriz de transição. Ele oculta a combinação de executado com tudo, exceto bateria e programa, deixando apenas as características que importam.

Matriz de transição mascarada

Depois de mascarar as características não úteis, as previsões da próxima palavra se tornam muito mais fortes. Quando a palavra bateria ocorre anteriormente na frase, a palavra depois de executado é prevista como abaixo com um peso de 1 e por favor com um peso de 0. O que era uma diferença de 25 por cento se tornou uma diferença de 100 por cento. Não há dúvida sobre qual palavra vem a seguir. O mesmo ocorre com a previsão forte para por favor quando o programa ocorre no início.

Este processo de mascaramento seletivo é a atenção destacada no título do artigo original sobre transformadores. Até agora, o que descrevemos é apenas uma aproximação de como a atenção é implementada no artigo. Ele captura os conceitos importantes, mas os detalhes são diferentes. Vamos fechar essa lacuna mais tarde.

Parada de Descanso e uma Saída
Parabéns por chegar até aqui, sagaz leitora. Você pode parar se quiser. O modelo de segunda ordem seletiva com saltos é uma maneira útil de pensar sobre o que os transformadores fazem, pelo menos no lado do decodificador. Ele captura, em uma primeira aproximação, o que os modelos de linguagem generativa como o GPT-3 da OpenAI estão fazendo. Não conta toda a história, mas representa a ideia central dela.

As próximas seções cobrem mais a lacuna entre essa explicação intuitiva e como os transformadores são implementados. Elas são amplamente motivadas por três considerações práticas.

Os computadores são especialmente bons em multiplicações de matrizes. Existe uma indústria inteira em torno da construção de hardware de computador especificamente para multiplicações de matrizes rápidas. Qualquer cálculo que possa ser expresso como uma multiplicação de matrizes pode ser tornado surpreendentemente eficiente. É um trem-bala. Se você conseguir colocar sua bagagem nele, ele te levará aonde você quiser ir bem rápido.
Cada etapa precisa ser diferenciável. Até agora, trabalhamos com exemplos de brinquedo e tivemos o luxo de escolher manualmente todas as probabilidades de transição e valores de máscara - os parâmetros do modelo. Na prática, eles precisam ser aprendidos por meio de retropropagação, *backpropagation*, o que depende de cada etapa de cálculo ser diferenciável. Isso significa que, para qualquer pequena mudança em um parâmetro, podemos calcular a mudança correspondente no erro ou perda do modelo.
O gradiente precisa ser suave e bem condicionado. A combinação de todas as derivadas para todos os parâmetros é o gradiente de perda. Na prática, fazer com que a retropropagação se comporte bem exige gradientes que sejam suaves, ou seja, a inclinação não muda muito rapidamente à medida que você dá pequenos passos em qualquer direção. Eles também se comportam muito melhor quando o gradiente está bem condicionado, ou seja, não é radicalmente maior em uma direção do que em outra. Se você imaginar uma função de perda como uma paisagem, o Grand Canyon seria um mal condicionado. Dependendo de você estar viajando ao longo do fundo ou subindo o lado, você terá inclinações muito diferentes para viajar. Em contraste, as colinas ondulantes do protetor de tela clássico do Windows teriam um gradiente bem condicionado.
Se a ciência de arquitetar redes neurais é criar blocos de construção diferenciáveis, a arte delas é empilhar as peças de maneira que o gradiente não mude muito rapidamente e seja mais ou menos da mesma magnitude em todas as direções.

Atenção como Multiplicação de Matrizes
Os pesos das características poderiam ser fáceis de construir contando quantas vezes cada transição de par de palavras/próxima palavra ocorre no treinamento, mas as máscaras de atenção não. Até este ponto, extraímos o vetor de máscara do nada. Como os transformadores encontram a máscara relevante importa. Seria natural usar algum tipo de tabela de consulta, mas agora estamos focando intensamente em expressar tudo como multiplicações de matrizes. Podemos usar o mesmo método de consulta que introduzimos acima empilhando os vetores de máscara para cada palavra em uma matriz e usando a representação one-hot da palavra mais recente para extrair a máscara relevante.

Consulta de máscara por multiplicação de matrizes

Na matriz mostrando a coleção de vetores de máscara, mostramos apenas aquele que estamos tentando extrair, para maior clareza.

Finalmente estamos chegando ao ponto em que podemos começar a relacionar isso com o artigo. Esta consulta de máscara é representada pelo termo QK^T na equação de atenção.

Equação de atenção destacando QKT

A consulta Q representa a característica de interesse e a matriz K representa a coleção de máscaras. Como é armazenada com máscaras em colunas, em vez de linhas, ela precisa ser transposta (com o operador T) antes da multiplicação. Quando terminarmos, faremos algumas modificações importantes nisso, mas neste nível ele captura o conceito de uma tabela de consulta diferenciável que os transformadores utilizam.

Modelo de Sequência de Segunda Ordem como Multiplicações de Matrizes
Outra etapa que fomos vagos até agora é a construção de matrizes de transição. Fomos claros sobre a lógica, mas não sobre como fazê-lo com multiplicações de matrizes.

Depois de obtermos o resultado de nossa etapa de atenção, um vetor que inclui a palavra mais recente e uma pequena coleção das palavras que a precederam, precisamos traduzir isso em características, cada uma das quais é um par de palavras. A máscara de atenção nos fornece a matéria-prima de que precisamos, mas não constrói essas características de pares de palavras. Para fazer isso, podemos usar uma rede neural totalmente conectada de camada única.

Para ver como uma camada de rede neural pode criar esses pares, vamos criar uma manualmente. Será artificialmente limpa e estilizada, e seus pesos não terão semelhança com os pesos na prática, mas demonstrará como a rede neural tem a expressividade necessária para construir essas características de dois pares de palavras. Para mantê-la pequena e limpa, focaremos nas três palavras atendidas deste exemplo: bateria, programa, executado.

Diagrama da camada de rede neural para criar características de várias palavras

No diagrama da camada acima, podemos ver como os pesos agem para combinar a presença e a ausência de cada palavra em uma coleção de características. Isso também pode ser expresso em forma de matriz.

Matriz de pesos para criar características de várias palavras

E pode ser calculado por uma multiplicação de matrizes com um vetor representando a coleção de palavras vistas até agora.

Cálculo da característica 'bateria, executado'

Os elementos bateria e executado são 1 e o elemento programa é 0. O elemento de viés é sempre 1, uma característica das redes neurais. Trabalhando na multiplicação de matrizes dá um 1 para o elemento representando bateria, executado e um -1 para o elemento representando programa, executado. Os resultados para o outro caso são semelhantes.

Cálculo da característica 'programa, executado'

A etapa final na criação dessas características de combinação de palavras é aplicar uma não linearidade de unidade de retificação linear (ReLU). O efeito disso é substituir qualquer valor negativo por um zero. Isso limpa ambos desses resultados para que representem a presença (com 1) ou ausência (com 0) de cada característica de combinação de palavras.

Com essa ginástica atrás de nós, finalmente temos um método baseado em multiplicação de matrizes para criar características de várias palavras. Embora eu tenha afirmado originalmente que essas consistem na palavra mais recente e em uma palavra anterior, uma inspeção mais atenta nesse método mostra que ele também pode construir outras características. Quando a matriz de criação de características é aprendida, em vez de codificada, outras estruturas podem ser aprendidas. Mesmo neste exemplo de brinquedo, não há nada que impeça a criação de uma combinação de três palavras como bateria, programa, executado. Se essa combinação ocorresse com frequência suficiente, provavelmente acabaria sendo representada. Não haveria como indicar em que ordem as palavras ocorreram (pelo menos ainda não), mas poderíamos usar absolutamente sua co-ocorrência para fazer previsões. Também seria possível usar combinações de palavras que ignoram a palavra mais recente, como bateria, programa. Essas e outros tipos de características provavelmente são criadas na prática, expondo a super simplificação que fiz quando afirmei que os transformadores são um modelo de sequência de segunda ordem seletiva com saltos. Há mais nuances do que isso e, agora, você pode ver exatamente qual é essa nuance. Esta não será a última vez que mudaremos a história para incorporar mais sutileza.

Nesta forma, a matriz de características de várias palavras está pronta para mais uma multiplicação de matrizes, a matriz de transição de segunda ordem com saltos que desenvolvemos acima. Todo o conjunto,

multiplicação da matriz de criação de características,
não linearidade ReLU e
multiplicação da matriz de transição
são as etapas de processamento propagação direta, *feedforward*, que são aplicadas após a aplicação da atenção. A equação 2 do artigo mostra essas etapas em uma formulação matemática concisa.

Equações por trás do bloco Feed Forward

A Figura 1 do diagrama de arquitetura do artigo mostra essas agrupadas como o bloco Feedforward.

Arquitetura do transformador mostrando o bloco Feed Forward

Conclusão da Sequência
Até agora, só falamos sobre a previsão da próxima palavra. Existem algumas peças que precisamos adicionar para que nosso decodificador gere uma sequência longa. A primeira é um prompt, algum texto de exemplo para dar ao transformador um início e contexto sobre o qual construir o resto da sequência. Ele é inserido no decodificador, a coluna à direita na imagem acima, onde está rotulado como "Saídas (deslocadas para a direita)". Escolher um prompt que gere sequências interessantes é uma arte em si, chamada engenharia de prompt. É também um ótimo exemplo de humanos modificando seu comportamento para apoiar algoritmos, em vez do contrário.

Depois que o decodificador recebe uma sequência parcial para começar, ele faz um passe para frente. O resultado final é um conjunto de distribuições de probabilidade previstas de palavras, uma distribuição de probabilidade para cada posição na sequência. Em cada posição, a distribuição mostra as probabilidades previstas para cada próxima palavra no vocabulário. Não nos importamos com as probabilidades previstas para cada palavra estabelecida na sequência. Elas já estão estabelecidas. O que realmente nos importa são as probabilidades previstas para a próxima palavra depois do final do prompt. Existem várias maneiras de abordar a escolha dessa palavra, mas a mais direta é chamada de gananciosa, escolhendo a palavra com a maior probabilidade.

A nova próxima palavra é então adicionada à sequência, substituída nas "Saídas" na parte inferior do decodificador, e o processo é repetido até que você se canse disso.

A peça que ainda não estamos prontos para descrever em detalhes é outra forma de mascaramento, garantindo que, quando o transformador faz previsões, ele só olhe para trás, não para frente. É aplicado no bloco rotulado como "Atenção Multi-Head Mascarada". Vamos revisitar isso mais tarde, quando pudermos ser mais claros sobre como é feito.

Embeddings
Como os descrevemos até agora, os transformadores são muito grandes. Para um tamanho de vocabulário N de, digamos, 50.000, a matriz de transição entre todos os pares de palavras e todas as possíveis próximas palavras teria 50.000 colunas e 50.000 quadrados (2,5 bilhões) de linhas, totalizando mais de 100 trilhões de elementos. Isso ainda é um estirão, mesmo para o hardware moderno.

Não é apenas o tamanho das matrizes que é o problema. Para construir um modelo de linguagem de transição estável, teríamos que fornecer dados de treinamento ilustrando cada sequência possível várias vezes, pelo menos. Isso excederia a capacidade até dos conjuntos de dados de treinamento mais ambiciosos.

Felizmente, existe uma solução alternativa para ambos esses problemas: embeddings.

Em uma representação one-hot de uma linguagem, existe um elemento de vetor para cada palavra. Para um vocabulário de tamanho N, esse vetor é um espaço N-dimensional. Cada palavra representa um ponto nesse espaço, uma unidade de distância da origem ao longo de um dos muitos eixos. Não consegui descobrir uma maneira ótima de desenhar um espaço de alta dimensão, mas há uma representação rudimentar dele abaixo.

Em um embedding, esses pontos de palavras são todos rearranjados (projetados, na terminologia de álgebra linear) em um espaço de dimensão inferior. A figura acima mostra como eles poderiam parecer em um espaço 2-dimensional, por exemplo. Agora, em vez de precisar de N números para especificar uma palavra, precisamos apenas de 2. Esses são as coordenadas (x, y) de cada ponto no novo espaço. Aqui está como poderia ser um embedding 2-dimensional para nosso exemplo de brinquedo, junto com as coordenadas de algumas das palavras.

Um bom embedding agrupa palavras com significados semelhantes. Um modelo que trabalha com um embedding aprende padrões no espaço embedded. Isso significa que o que quer que ele aprenda a fazer com uma palavra é automaticamente aplicado a todas as palavras próximas a ela. Isso tem a vantagem adicional de reduzir a quantidade de dados de treinamento necessários. Cada exemplo dá um pouco de aprendizado que é aplicado em toda uma vizinhança de palavras.

Nesta ilustração, tentei mostrar isso colocando componentes importantes em uma área (bateria, log, programa), preposições em outra (abaixo, fora) e verbos perto do centro (verificar, encontrar, executado). Em um embedding real, os agrupamentos podem não ser tão claros ou intuitivos, mas o conceito subjacente é o mesmo. A distância é pequena entre palavras que se comportam de maneira semelhante.

Um embedding reduz o número de parâmetros necessários em uma quantidade tremenda. No entanto, quanto menor o número de dimensões no espaço embedded, mais informações sobre as palavras originais são descartadas. A riqueza de uma linguagem ainda exige bastante espaço para dispor todos os conceitos importantes de maneira que não pisem uns nos outros. Ao escolher o tamanho do espaço embedded, trocamos a carga computacional pela precisão do modelo.

Provavelmente não te surpreenderá saber que projetar palavras de sua representação one-hot para um espaço embedded envolve uma multiplicação de matrizes. A projeção é o que as matrizes fazem de melhor. Começando com uma matriz one-hot que tem uma linha e N colunas, e passando para um espaço embedded de duas dimensões, a matriz de projeção terá N linhas e duas colunas, como mostrado aqui.

Uma matriz de projeção descrevendo um embedding

Este exemplo mostra como um vetor one-hot, representando, por exemplo, bateria, extrai a linha associada a ele, que contém as coordenadas da palavra no espaço embedded. Para tornar o relacionamento mais claro, os zeros no vetor one-hot são ocultados, assim como todas as outras linhas que não são extraídas da matriz de projeção. A matriz de projeção completa é densa, cada linha contendo as coordenadas da palavra com a qual está associada.

As matrizes de projeção podem converter a coleção original de vetores de vocabulário one-hot em qualquer configuração em um espaço de qualquer dimensionalidade que você queira. O maior truque é encontrar uma projeção útil, que tenha palavras semelhantes agrupadas e que tenha dimensões suficientes para espalhá-las. Existem alguns embeddings pré-computados decentes para idiomas comuns, como o inglês. Também, como tudo mais no transformador, ele pode ser aprendido durante o treinamento.

Na Figura 1 do diagrama de arquitetura do artigo original, aqui está onde o embedding ocorre.

Arquitetura do transformador mostrando o bloco de embedding

Codificação Posicional
Até este ponto, assumimos que as posições das palavras são ignoradas, pelo menos para qualquer palavra que venha antes da palavra mais recente. Agora vamos corrigir isso usando embeddings posicionais.

Existem várias maneiras pelas quais as informações de posição podem ser introduzidas em nossa representação embedded de palavras, mas a maneira como foi feita no transformador original foi adicionar uma ondulação circular.

A codificação posicional introduz uma ondulação circular

A posição da palavra no espaço embedded atua como o centro de um círculo. Uma perturbação é adicionada a ele, dependendo de onde ele cai na ordem da sequência de palavras. Para cada posição, a palavra é movida na mesma distância, mas em um ângulo diferente, resultando em um padrão circular à medida que você se move pela sequência. Palavras que estão próximas umas das outras na sequência têm perturbações semelhantes, mas palavras que estão distantes são perturbadas em direções diferentes.

Como um círculo é uma figura bidimensional, representar uma ondulação circular requer modificar duas dimensões do espaço embedded. Se o espaço embedded consiste em mais de duas dimensões (o que quase sempre acontece), a ondulação circular é repetida em todos os outros pares de dimensões, mas com diferentes frequências angulares, ou seja, ela varre diferentes números de rotações em cada caso. Em alguns pares de dimensões, a ondulação varrerá muitas rotações do círculo. Em outros pares, ela varrerá apenas uma fração pequena de uma rotação. A combinação de todas essas ondulações circulares de diferentes frequências dá uma boa representação da posição absoluta de uma palavra dentro da sequência.

Ainda estou desenvolvendo minha intuição sobre por que isso funciona. Parece adicionar informações de posição à mistura de uma maneira que não interrompe os relacionamentos aprendidos entre palavras e atenção. Para um mergulho mais profundo na matemática e nas implicações, recomendo o tutorial de codificação posicional de Amirhossein Kazemnejad.

Nos blocos de arquitetura canônica, esses blocos mostram a geração do código de posição e sua adição às palavras embedded.

Arquitetura do transformador mostrando a codificação posicional

De-embeddings
Embeddings tornam as palavras vastamente mais eficientes para trabalhar, mas depois que a festa acaba, elas precisam ser convertidas de volta em palavras do vocabulário original. O de-embedding é feito da mesma maneira que os embeddings, com uma projeção de um espaço para outro, ou seja, uma multiplicação de matrizes.

A matriz de de-embedding tem a mesma forma que a matriz de embedding, mas com o número de linhas e colunas invertidas. O número de linhas é a dimensionalidade do espaço do qual estamos convertendo. Neste exemplo, é o tamanho do nosso espaço embedded, dois. O número de colunas é a dimensionalidade do espaço para o qual estamos convertendo - o tamanho da representação one-hot do vocabulário completo, 13 em nosso exemplo.

A transformação de de-embedding

Os valores em uma boa matriz de de-embedding não são tão diretos de ilustrar quanto os da matriz de embedding, mas o efeito é semelhante. Quando um vetor embedded representando, digamos, a palavra programa é multiplicado pela matriz de de-embedding, o valor na posição correspondente é alto. No entanto, devido à maneira como a projeção para espaços de dimensão mais alta funciona, os valores associados às outras palavras não serão zero. As palavras mais próximas de programa no espaço embedded também terão valores médio-altos. Outras palavras terão valores próximos de zero. E provavelmente haverá muitas palavras com valores negativos. O vetor de resultado no espaço do vocabulário não será mais one-hot ou esparsa. Será denso, com quase todos os valores diferentes de zero.

Vetor de resultado denso representativo do de-embedding

Tudo bem. Podemos recriar o vetor one-hot escolhendo a palavra associada ao valor mais alto. Essa operação também é chamada de argmax, o argumento (elemento) que dá o valor máximo. Esta é a maneira de fazer a conclusão da sequência de forma gananciosa, como mencionado acima. É uma ótima primeira abordagem, mas podemos fazer melhor.

Se um embedding mapeia muito bem para várias palavras, talvez não queiramos escolher a melhor todas as vezes. Pode ser apenas uma fração melhor do que as outras e adicionar um toque de variedade pode tornar o resultado mais interessante. Além disso, às vezes é útil olhar algumas palavras à frente e considerar todas as direções que a frase pode tomar antes de se estabelecer em uma escolha final. Para fazer isso, precisamos primeiro converter nossos resultados de de-embedding em uma distribuição de probabilidade.

Softmax
A função argmax é "dura" no sentido de que o valor mais alto vence, mesmo que seja apenas infinitesimalmente maior que os outros. Se quisermos entreter várias possibilidades ao mesmo tempo, é melhor ter uma função de máximo "suave", que obtemos do softmax. Para obter o softmax do valor x em um vetor, divida o exponencial de x, e^x, pela soma dos exponenciais de todos os valores no vetor.

O softmax é útil aqui por três razões. Primeiro, ele converte nosso vetor de resultados de de-embedding de um conjunto arbitrário de valores para uma distribuição de probabilidade. Como probabilidades, fica mais fácil comparar a probabilidade de diferentes palavras serem selecionadas e até comparar a probabilidade de sequências de múltiplas palavras, se quisermos olhar mais adiante no futuro.

Em segundo lugar, ele afinam o campo perto do topo. Se uma palavra pontua claramente mais alto que as outras, o softmax exagerará essa diferença, fazendo com que pareça quase um argmax, com o valor vencedor próximo de um e todos os outros próximos de zero. No entanto, se houver várias palavras que saem no topo, ele as preservará todas como altamente prováveis, em vez de artificialmente esmagar resultados de segundo lugar próximos.

Em terceiro lugar, o softmax é diferenciável, o que significa que podemos calcular quanto cada elemento dos resultados mudará, dada uma pequena mudança em qualquer um dos elementos de entrada. Isso nos permite usá-lo com retropropagação para treinar nosso transformador.

Se você sentir vontade de aprofundar seu entendimento sobre softmax (ou se tiver dificuldade para dormir à noite), aqui está um post mais completo sobre isso.

Juntos, a transformação de de-embedding (mostrada como o bloco Linear abaixo) e uma função softmax completam o processo de de-embedding.

Arquitetura do transformador mostrando o de-embedding

Atenção Multi-Head
Agora que fizemos as pazes com os conceitos de projeções (multiplicações de matrizes) e espaços (tamanhos de vetores), podemos revisitar o mecanismo central de atenção com vigor renovado. Vai ajudar a esclarecer o algoritmo se pudermos ser mais específicos sobre a forma de nossas matrizes em cada estágio. Existe uma lista curta de números importantes para isso.

N: tamanho do vocabulário. 13 em nosso exemplo. Tipicamente na casa das dezenas de milhares.
n: comprimento máximo da sequência. 12 em nosso exemplo. Algo como algumas centenas no artigo. (Eles não especificam.) 2048 no GPT-3.
d_model: número de dimensões no espaço embedded usado em todo o modelo. 512 no artigo.
A matriz de entrada original é construída obtendo cada uma das palavras da frase em sua representação one-hot e empilhando-as de modo que cada um dos vetores one-hot seja sua própria linha. A matriz de entrada resultante tem n linhas e N colunas, que podemos abreviar como [n x N].

A multiplicação de matrizes altera as formas das matrizes

Como ilustramos antes, a matriz de embedding tem N linhas e d_model colunas, que podemos abreviar como [N x d_model]. Quando multiplicamos duas matrizes, o resultado obtém seu número de linhas da primeira matriz e seu número de colunas da segunda. Isso dá à matriz de sequência de palavras embedded a forma [n x d_model].

Podemos acompanhar as mudanças na forma da matriz através do transformador como uma maneira de acompanhar o que está acontecendo. Depois do embedding inicial, a codificação posicional é aditiva, em vez de uma multiplicação, então ela não altera a forma das coisas. Então, a sequência de palavras embedded entra nas camadas de atenção e sai do outro lado na mesma forma. (Vamos voltar aos funcionamentos internos dessas em um segundo.) Finalmente, o de-embedding restaura a matriz à sua forma original, oferecendo uma probabilidade para cada palavra no vocabulário em cada posição na sequência.

Formas das matrizes ao longo do modelo do transformador

Por que Precisamos de Mais de uma Cabeça de Atenção
Finalmente, chegou a hora de confrontar algumas das suposições simplistas que fizemos durante nossa primeira explicação do mecanismo de atenção. As palavras são representadas como vetores densos embedded, em vez de vetores one-hot. A atenção não é apenas 1 ou 0, ligado ou desligado, mas também pode estar em qualquer lugar entre eles. Para obter os resultados a caírem entre 0 e 1, usamos o truque do softmax novamente. Ele tem a vantagem dupla de forçar todos os valores a estarem em nosso intervalo de atenção [0, 1] e ajuda a enfatizar o valor mais alto, enquanto agressivamente esmagando o menor. É o comportamento de quase-argmax diferenciável que aproveitamos antes ao interpretar o resultado final do modelo.

Uma consequência complicada de colocar uma função softmax na atenção é que ela tenderá a se concentrar em um único elemento. Isso é uma limitação que não tínhamos antes. Às vezes, é útil manter várias das palavras anteriores em mente ao prever a próxima e o softmax acabou de nos roubar isso. Esta é uma dificuldade para o modelo.

A solução é ter várias instâncias de atenção, ou cabeças, funcionando ao mesmo tempo. Isso permite que o transformador considere várias palavras anteriores simultaneamente ao prever a próxima. Isso traz de volta o poder que tínhamos antes de colocarmos o softmax na jogada.

Infelizmente, fazer isso realmente aumenta a carga computacional. Calcular a atenção já era a maior parte do trabalho e acabamos de multiplicá-lo pelo número de cabeças que queremos usar. Para contornar isso, podemos reutilizar o truque de projetar tudo em um espaço embedded de dimensão inferior. Isso reduz o tamanho das matrizes envolvidas, o que diminui drasticamente o tempo de computação. O dia é salvo.

Para ver como isso se desenrola, podemos continuar acompanhando a forma da matriz através dos ramos e entrelaçamentos dos blocos de atenção multi-head. Para isso, precisamos de três números adicionais.

d_k: dimensões no espaço embedded usado para consultas e chaves. 64 no artigo.
d_v: dimensões no espaço embedded usado para valores. 64 no artigo.
h: o número de cabeças. 8 no artigo.
Arquitetura do transformador mostrando a atenção multi-head

A sequência [n x d_model] de palavras embedded serve como base para tudo o que se segue. Em cada caso, existe uma matriz, Wv, Wq e Wk, (todas mostradas de forma pouco útil como blocos "Linear" no diagrama de arquitetura) que transforma a sequência original de palavras embedded na matriz de valores, V, na matriz de consultas, Q, e na matriz de chaves, K. K e Q têm a mesma forma, [n x d_k], mas V pode ser diferente, [n x d_v]. Confunde um pouco o fato de d_k e d_v serem os mesmos no artigo, mas eles não precisam ser. Um aspecto importante dessa configuração é que cada cabeça de atenção tem suas próprias transformações Wv, Wq e Wk. Isso significa que cada cabeça pode ampliar e focar nas partes do espaço embedded que deseja, e pode ser diferente do que cada uma das outras cabeças está focando.

O resultado de cada cabeça de atenção tem a mesma forma que V. Agora temos o problema de h vetores de resultado diferentes, cada um atendendo a diferentes elementos da sequência. Para combinar esses em um único vetor gigante, exploramos os poderes da álgebra linear e simplesmente concatenamos todos esses resultados em uma matriz gigante $[n \times h \times d_v]$. Em seguida, para garantir que ele termine na mesma forma com que começou, usamos mais uma transformação com a forma $[h \times d_v \times d_model]$.

Aqui está tudo isso, dito de forma concisa.

Equação de atenção multi-head do artigo

Atenção de Cabeça Única Revisitada
Já percorremos uma ilustração conceitual da atenção acima. A implementação real é um pouco mais confusa, mas nossa intuição inicial ainda é útil. As consultas e as chaves já não são fáceis de inspecionar e interpretar porque são todas projetadas em seus próprios subespaços idiomáticos. Em nossa ilustração conceitual, uma linha na matriz de consultas representava um ponto no espaço do vocabulário que, graças à representação one-hot, representava uma e apenas uma palavra. Em sua forma embedded, uma linha na matriz de consultas representa um ponto no espaço embedded, que estará próximo de um grupo de palavras com significados e usos semelhantes. A ilustração conceitual mapeava uma palavra de consulta para um conjunto de chaves, que, por sua vez, filtrava todos os valores que não estavam sendo atendidos. Cada cabeça de atenção na implementação real mapeia uma palavra de consulta para um ponto em outro espaço embedded de dimensão inferior. O resultado disso é que a atenção se torna um relacionamento entre grupos de palavras, em vez de entre palavras individuais. Ele aproveita as semelhanças semânticas (proximidade no espaço embedded) para generalizar o que aprendeu sobre palavras semelhantes.

Acompanhar a forma das matrizes através do cálculo da atenção ajuda a rastrear o que ela está fazendo.

Arquitetura do transformador mostrando a atenção de cabeça única

As matrizes de consultas e chaves, Q e K, entram ambas com a forma [n x d_k]. Graças a K ser transposta antes da multiplicação, o resultado de Q K^T dá uma matriz de [n x d_k] * [d_k x n] = [n x n]. Dividir cada elemento dessa matriz pela raiz quadrada de d_k foi demonstrado que mantém a magnitude dos valores sem crescer descontroladamente e ajuda a retropropagação a se comportar bem. O softmax, como mencionamos, força o resultado a se aproximar de um argmax, tendendo a focar a atenção em um elemento da sequência mais do que no resto. Nesta forma, a matriz de atenção [n x n] mapeia aproximadamente cada elemento da sequência para outro elemento da sequência, indicando o que ele deve observar para obter o contexto mais relevante para prever o próximo elemento. É um filtro que, por fim, é aplicado à matriz de valores V, deixando apenas uma coleção dos valores atendidos. Isso tem o efeito de ignorar a grande maioria do que veio antes na sequência e lança um holofote sobre o único elemento anterior que é mais útil estar ciente.

A equação de atenção

Uma parte complicada de entender esse conjunto de cálculos é manter em mente que ele está calculando a atenção para cada elemento de nossa sequência de entrada, para cada palavra em nossa frase, não apenas para a palavra mais recente. Ele também está calculando a atenção para palavras anteriores. Não nos importamos muito com essas porque suas próximas palavras já foram previstas e estabelecidas. Ele também está calculando a atenção para palavras futuras. Essas ainda não têm muita utilidade porque estão muito à frente e suas predecessoras imediatas ainda não foram escolhidas. Mas existem caminhos indiretos pelos quais esses cálculos podem afetar a atenção para a palavra mais recente, então os incluímos todos. É apenas que, quando chegarmos ao final e calcularmos as probabilidades de palavras para cada posição na sequência, descartaremos a maioria delas e prestaremos atenção apenas na próxima palavra.

O bloco de máscara aplica a restrição de que, pelo menos para esta tarefa de conclusão de sequência, não podemos olhar para o futuro. Ele evita introduzir quaisquer artefatos estranhos de palavras futuras imaginárias. É cru e eficaz - manualmente define a atenção paga a todas as palavras após a posição atual como negativa infinita. Na Annotated Transformer, um companheiro imensamente útil do artigo mostrando a implementação em Python linha por linha, a matriz de máscara é visualizada. Células roxas mostram onde a atenção é proibida. Cada linha corresponde a um elemento na sequência. A primeira linha é permitida atender a si mesma (o primeiro elemento) e a nada depois. A última linha é permitida atender a si mesma (o elemento final) e a tudo o que vem antes. A máscara é uma matriz [n x n]. Ela é aplicada não com uma multiplicação de matrizes, mas com uma multiplicação elemento por elemento mais direta. Isso tem o efeito de entrar manualmente na matriz de atenção e definir todos os elementos roxos da máscara como infinito negativo.

Uma máscara de atenção para conclusão de sequência

Outra diferença importante em como a atenção é implementada é que ela usa a ordem na qual as palavras são apresentadas a ela na sequência e representa a atenção não como um relacionamento palavra-palavra, mas como um relacionamento posição-posição. Isso é evidente em sua forma [n x n]. Ele mapeia cada elemento da sequência, indicado pelo índice da linha, para algum outro elemento (ou elementos) da sequência, indicado pelo índice da coluna. Isso nos ajuda a visualizar e interpretar o que ele está fazendo mais facilmente, já que está operando no espaço embedded. Somos poupados do passo extra de encontrar palavras próximas no espaço embedded para representar os relacionamentos entre consultas e chaves.

Conexão de Salto
A atenção é a parte mais fundamental do que os transformadores fazem. É o mecanismo central e já o percorremos em um nível bastante concreto. Tudo daqui para frente é a canalização necessária para fazer isso funcionar bem. É o resto do arreio que permite que a atenção puxe nossas cargas pesadas.

Uma peça que ainda não explicamos são as conexões de salto. Elas ocorrem em torno dos blocos de Atenção Multi-Head e em torno dos blocos de Feed Forward elemento a elemento nos blocos rotulados como "Add e Norm". Nas conexões de salto, uma cópia da entrada é adicionada à saída de um conjunto de cálculos. As entradas para o bloco de atenção são adicionadas de volta à sua saída. As entradas para o bloco de feed forward elemento a elemento são adicionadas às suas saídas.

Arquitetura do transformador mostrando os blocos Add e Norm

As conexões de salto servem a dois propósitos.

O primeiro é que elas ajudam a manter o gradiente suave, o que é uma grande ajuda para a retropropagação. A atenção é um filtro, o que significa que, quando está funcionando corretamente, bloqueará a maioria do que tentar passar por ela. O resultado disso é que pequenas mudanças em muitas das entradas podem não produzir muita mudança nas saídas se acontecerem de cair em canais que estão bloqueados. Isso produz pontos mortos no gradiente onde ele é plano, mas ainda assim longe do fundo de um vale. Esses pontos de sela e cristas são um grande tropeço para a retropropagação. As conexões de salto ajudam a suavizar isso. No caso da atenção, mesmo que todos os pesos fossem zero e todas as entradas fossem bloqueadas, uma conexão de salto adicionaria uma cópia das entradas aos resultados e garantiria que pequenas mudanças em qualquer uma das entradas ainda resultassem em mudanças perceptíveis no resultado. Isso mantém o gradiente descendente longe de ficar preso longe de uma boa solução.

As conexões de salto se tornaram populares desde os dias do classificador de imagens ResNet. Agora, elas são uma característica padrão nas arquiteturas de redes neurais. Visualmente, podemos ver o efeito que as conexões de salto têm comparando redes com e sem elas. A figura abaixo deste artigo mostra um ResNet com e sem conexões de salto. As inclinações das superfícies de perda de função são muito mais moderadas e uniformes quando as conexões de salto são usadas. Se você sentir vontade de mergulhar mais fundo em como elas funcionam e por quê, há um tratamento mais aprofundado neste post.

Comparação das superfícies de perda com e sem conexões de salto

O segundo propósito das conexões de salto é específico dos transformadores - preservar a sequência de entrada original. Mesmo com muitas cabeças de atenção, não há garantia de que uma palavra atenderá à sua própria posição. É possível que o filtro de atenção esqueça completamente a palavra mais recente em favor de observar todas as palavras anteriores que possam ser relevantes. Uma conexão de salto pega a palavra original e a adiciona manualmente de volta ao sinal, de modo que não haja como ela ser descartada ou esquecida. Essa fonte de robustez pode ser uma das razões para o bom comportamento dos transformadores em tantas tarefas variadas de conclusão de sequências.

Normalização de Camada
A normalização é uma etapa que combina bem com conexões de salto. Não há necessidade de que elas sempre estejam juntas, mas elas fazem seu melhor trabalho quando colocadas após um grupo de cálculos, como atenção ou uma rede neural feed forward.

A versão curta da normalização de camada é que os valores da matriz são deslocados para ter uma média de zero e dimensionados para ter um desvio padrão de um.

Várias distribuições sendo normalizadas

A versão mais longa é que, em sistemas como transformadores, onde há muitas peças móveis e algumas delas são algo diferente de multiplicações de matrizes (como operadores softmax ou unidades de retificação linear), importa o quão grandes os valores são e como eles estão equilibrados entre positivo e negativo. Se tudo for linear, você pode dobrar todas as suas entradas e suas saídas serão duas vezes maiores e tudo funcionará bem. Não é assim com redes neurais. Elas são inerentemente não lineares, o que as torna muito expressivas, mas também sensíveis às magnitudes e distribuições dos sinais. A normalização é uma técnica que se mostrou útil em manter uma distribuição consistente de valores de sinal em cada etapa ao longo de redes neurais de muitas camadas. Ela encoraja a convergência dos valores dos parâmetros e geralmente resulta em um desempenho muito melhor.

Minha coisa favorita sobre a normalização é que, além das explicações de alto nível que acabei de dar, ninguém tem certeza absoluta do porquê ela funciona tão bem. Se você quiser descer um pouco mais fundo nessa toca de coelho, escrevi um post mais detalhado sobre a normalização de lote, um primo próximo da normalização de camada usada nos transformadores.

Múltiplas Camadas
Enquanto estávamos lançando os alicerces acima, mostramos que uma camada de atenção e um bloco feed forward com pesos cuidadosamente escolhidos eram suficientes para fazer um modelo de linguagem decente. Na maioria dos casos, os pesos eram zeros em nossos exemplos, alguns deles eram uns e todos foram escolhidos manualmente. Quando treinando a partir de dados brutos, não teremos esse luxo. No início, os pesos são todos escolhidos aleatoriamente, a maioria deles está próxima de zero e os poucos que não são provavelmente não são os que precisamos. É um longo caminho do que precisa ser para que nosso modelo funcione bem.

O gradiente descendente estocástico por meio da retropropagação pode fazer coisas bastante impressionantes, mas ele depende muito da sorte. Se houver apenas um caminho para a resposta correta, apenas uma combinação de pesos necessária para que a rede funcione bem, então é improvável que ele encontre seu caminho. Mas se houver muitos caminhos para uma boa solução, as chances são muito maiores de que o modelo chegue lá.

Ter apenas uma camada de atenção (apenas um bloco de atenção multi-head e um bloco feed forward) permite apenas um caminho para um bom conjunto de parâmetros de transformador. Cada elemento de cada matriz precisa encontrar seu caminho para o valor certo para fazer as coisas funcionarem bem. É frágil e quebradiço, provavelmente ficará preso em uma solução muito longe do ideal, a menos que as suposições iniciais para os parâmetros sejam muito, muito sortudas.

A maneira como os transformadores contornam esse problema é tendo múltiplas camadas de atenção, cada uma usando a saída da anterior como sua entrada. O uso de conexões de salto torna o pipeline geral robusto a camadas individuais falhando ou dando resultados estranhos. Tendo múltiplas, significa que há outras esperando para assumir a liderança. Se uma falhar ou, de alguma forma, não conseguir cumprir seu potencial, haverá outra a jusante que terá outra chance de fechar a lacuna ou corrigir o erro. O artigo mostrou que mais camadas resultaram em melhor desempenho, embora a melhoria tenha se tornado marginal após 6.

Outra maneira de pensar em múltiplas camadas é como uma linha de montagem de esteira transportadora. Cada bloco de atenção e bloco feed forward tem a chance de retirar entradas da linha, calcular matrizes de atenção úteis e fazer previsões da próxima palavra. Quaisquer resultados que produzam, úteis ou não, são adicionados de volta à esteira e passados para a próxima camada.

Transformador redesenhado como uma esteira transportadora

Isso contrasta com a descrição tradicional de redes neurais profundas como "profundas". Graças às conexões de salto, camadas sucessivas não fornecem abstrações cada vez mais sofisticadas, mas sim redundância. Qualquer oportunidade de focar a atenção e criar características úteis e fazer previsões precisas que foram perdidas em uma camada pode sempre ser capturada pela próxima. As camadas se tornam trabalhadores na linha de montagem, onde cada um faz o que pode, mas não se preocupa em capturar todas as peças, porque o próximo trabalhador capturará as que perderem.

Pilha de Decodificação
Até agora, cuidadosamente ignoramos a pilha de codificação (o lado esquerdo da arquitetura do transformador) em favor da pilha de decodificação (o lado direito). Vamos corrigir isso em alguns parágrafos. Mas vale a pena notar que o decodificador sozinho é bastante útil.

Como esboçamos na descrição da tarefa de conclusão de sequência, o decodificador pode completar sequências parciais e estendê-las pelo tempo que você quiser. A OpenAI criou o modelo generativo de pré-treinamento (GPT) família de modelos para fazer exatamente isso. A arquitetura que descrevem neste relatório deve parecer familiar. É um transformador com a pilha de codificação e todas as suas conexões removidas cirurgicamente. O que resta é uma pilha de decodificação de 12 camadas.

Arquitetura da família de modelos GPT

Sempre que você encontrar um modelo generativo, como BERT, ELMo ou Copilot, provavelmente estará vendo a metade do decodificador de um transformador em ação.

Pilha de Codificação
Quase tudo o que aprendemos sobre o decodificador também se aplica ao codificador. A maior diferença é que não há previsões explícitas sendo feitas no final que possamos usar para julgar a correção ou incorreção de seu desempenho. Em vez disso, o produto final de uma pilha de codificação é decepcionantemente abstrato - uma sequência de vetores em um espaço embedded. É descrito como uma representação semântica pura da sequência, divorciada de qualquer idioma ou vocabulário específico, mas isso parece romanticamente exagerado para mim. O que sabemos com certeza é que é um sinal útil para comunicar intenção e significado à pilha de decodificação.

Ter uma pilha de codificação abre todo o potencial dos transformadores; em vez de apenas gerar sequências, eles podem agora traduzir (ou transformar) a sequência de um idioma para outro. O treinamento em uma tarefa de tradução é diferente do treinamento em uma tarefa de conclusão de sequência. Os dados de treinamento exigem tanto uma sequência na língua de origem quanto uma sequência correspondente na língua de destino. A pilha de codificação completa (sem mascaramento desta vez, pois assumimos que temos a frase completa antes de criar uma tradução) e o resultado, a saída da camada de codificação final, é fornecido como entrada para cada uma das camadas de decodificação. Em seguida, a geração de sequência no decodificador prossegue como antes, mas desta vez sem um prompt para iniciar.

Atenção Cruzada
A etapa final para colocar o transformador completo em funcionamento é a conexão entre as pilhas de codificação e decodificação, o bloco de atenção cruzada. Já economizamos isso para o final e, graças aos alicerces que lançamos, não resta muito para explicar.

A atenção cruzada funciona da mesma maneira que a autoatenção, com a exceção de que a matriz de chaves K e a matriz de valores V são baseadas na saída da camada de codificação final, em vez da saída da camada de decodificação anterior. A matriz de consultas Q ainda é calculada a partir dos resultados da camada de decodificação anterior. Este é o canal pelo qual as informações da sequência de origem encontram seu caminho na sequência de destino e direcionam sua criação na direção correta. É interessante notar que a mesma sequência embedded de origem é fornecida a cada camada do decodificador, apoiando a noção de que camadas sucessivas fornecem redundância e todas estão cooperando para realizar a mesma tarefa.

Arquitetura do transformador mostrando o bloco de atenção cruzada

Tokenização
Conseguimos passar por todo o transformador! Cobrimos em detalhes suficientes para que não haja mais caixas pretas misteriosas. Existem alguns detalhes de implementação que não detalhamos. Você precisaria saber sobre eles para construir uma versão funcional para si mesmo. Esses últimos detalhes não são tanto sobre como os transformadores funcionam, mas sobre fazer com que as redes neurais se comportem bem. O Annotated Transformer ajudará você a preencher essas lacunas.

Ainda não terminamos, no entanto. Ainda existem algumas coisas importantes a dizer sobre como representamos os dados para começar. Este é um tópico que está próximo do meu coração, mas é fácil negligenciar. Não se trata tanto do poder do algoritmo, mas sim de interpretar os dados de maneira pensativa e entender o que eles significam.

Mencionamos de passagem que um vocabulário poderia ser representado por um vetor one-hot de alta dimensão, com um elemento associado a cada palavra. Para fazer isso, precisamos saber exatamente quantas palavras vamos representar e quais são elas.

Uma abordagem ingênua é fazer uma lista de todas as palavras possíveis, como poderíamos encontrar no Dicionário Webster. Para o idioma inglês, isso nos daria várias dezenas de milhares, o número exato dependendo do que escolhemos incluir ou excluir. Mas isso é uma simplificação excessiva. A maioria das palavras tem várias formas, incluindo plurais, possessivos e conjugações. As palavras podem ter grafias alternativas. E a menos que seus dados tenham sido cuidadosamente limpos, eles conterão erros de digitação de todos os tipos. Isso nem sequer toca nas possibilidades abertas pelo uso livre de texto, neologismos, gírias, jargões e o vasto universo do Unicode. Uma lista exaustiva de todas as palavras possíveis seria inexequivelmente longa.

Uma solução alternativa razoável seria permitir que caracteres individuais servissem como os blocos de construção, em vez de palavras. Uma lista exaustiva de caracteres está bem dentro da capacidade que temos para computar. No entanto, existem alguns problemas com isso. Depois de transformarmos os dados em um espaço embedded, assumimos que a distância nesse espaço tem uma interpretação semântica, ou seja, presumimos que pontos que caem próximos uns dos outros têm significados semelhantes e pontos que estão distantes significam algo muito diferente. Isso nos permite estender implicitamente o que aprendemos sobre uma palavra para seus vizinhos imediatos, uma suposição da qual o transformador extrai parte de sua capacidade de generalização.

No nível de caracteres individuais, há muito pouco conteúdo semântico. Existem algumas palavras de um caractere no idioma inglês, por exemplo, mas não muitas. Emojis são a exceção a isso, mas eles não são o conteúdo principal da maioria dos conjuntos de dados que estamos analisando. Isso nos deixa em uma situação infeliz de ter um espaço embedded inútil.

Poderia ser possível contornar isso teoricamente, se pudéssemos olhar para combinações ricas de caracteres para construir sequências semanticamente úteis, como palavras, radicais de palavras ou pares de palavras. Infelizmente, as características que os transformadores criam internamente se comportam mais como uma coleção de pares de entrada do que uma sequência ordenada de entradas. Isso significa que a representação de uma palavra seria uma coleção de pares de caracteres, sem sua ordem fortemente representada. O transformador seria forçado a lidar continuamente com anagramas, dificultando muito seu trabalho. E, de fato, experimentos com representações de nível de caractere mostraram que os transformadores não funcionam bem com elas.

Codificação de Pares de Bytes
Felizmente, existe uma solução elegante para isso. Chama-se codificação de pares de bytes. Começando com a representação de nível de caractere, cada caractere recebe seu próprio código, seu próprio byte exclusivo. Em seguida, após digitalizar alguns dados representativos, o par de bytes mais comum é agrupado e recebe um novo byte, um novo código. Esse novo código é substituído de volta nos dados e o processo é repetido.

Códigos representando pares de caracteres podem ser combinados com códigos representando outros caracteres ou pares de caracteres para obter novos códigos representando sequências de caracteres mais longas. Não há limite para o comprimento da sequência de caracteres que um código pode representar. Eles crescerão o quanto for necessário para representar sequências de caracteres comuns repetidas. A parte legal da codificação de pares de bytes é que ela infere quais sequências longas de caracteres aprender a partir dos dados, em vez de aprender de maneira simplória a representar todas as sequências possíveis. Ela aprende a representar palavras longas como transformador com um único código de byte, mas não desperdiçaria um código em uma sequência arbitrária de comprimento semelhante, como ksowjmckder. E como retém todos os códigos de byte para seus blocos de construção de caracteres individuais, ela ainda pode representar grafias estranhas, novas palavras e até idiomas estrangeiros.

Quando você usa a codificação de pares de bytes, você decide um tamanho de vocabulário e ela continua criando novos códigos até atingir esse tamanho. O tamanho do vocabulário precisa ser grande o suficiente para que as sequências de caracteres se tornem longas o suficiente para capturar o conteúdo semântico do texto. Elas precisam significar algo. Então, elas estarão suficientemente ricas para alimentar os transformadores.

Depois que um codificador de pares de bytes é treinado ou emprestado, podemos usá-lo para pré-processar nossos dados antes de alimentá-los no transformador. Isso quebra o fluxo ininterrupto de texto em uma sequência de pedaços distintos (a maioria dos quais, esperamos, será reconhecível como palavras) e fornece um código conciso para cada um. Este é o processo chamado de tokenização.

Entrada de Áudio
Agora, lembre-se de que nosso objetivo original quando começamos toda essa aventura era traduzir o sinal de áudio de um comando falado para uma representação de texto. Até agora, todos os nossos exemplos foram elaborados com a suposição de que estávamos trabalhando com caracteres e palavras da linguagem escrita. Podemos estender isso também para o áudio, mas isso exigirá uma incursão ainda mais ousada no pré-processamento de sinais.

As informações nos sinais de áudio se beneficiam de um pré-processamento intenso para extrair as partes que nossos ouvidos e cérebros usam para entender a fala. O método é chamado de filtragem cepstral de Mel-frequência e é tão barroco quanto o nome sugere. Aqui está um tutorial bem ilustrado, se você quiser mergulhar nos detalhes fascinantes.

Quando o pré-processamento é concluído, o áudio bruto é transformado em uma sequência de vetores, onde cada elemento representa a mudança de atividade de áudio em uma faixa de frequência específica. É denso (nenhum elemento é zero) e cada elemento é um valor real.

Por outro lado, tratar cada vetor como uma "palavra" ou token para o transformador é estranho porque cada um é único. É extremamente improvável que a mesma combinação exata de valores de vetor ocorra duas vezes, porque há tantas combinações sutilmente diferentes de sons. Nossas suposições anteriores de representação one-hot e codificação de pares de bytes não são de grande ajuda.

O truque aqui é perceber que vetores densos de valores reais como este são o que os transformadores adoram. Eles se alimentam desse formato. Para usá-los, podemos usar os resultados do pré-processamento cepstral como faríamos com as palavras embedded de um exemplo de texto. Isso economiza os passos de tokenização e embedding.

Vale a pena notar que podemos fazer isso com qualquer outro tipo de dado que quisermos também. Muitos dados gravados vêm na forma de uma sequência de vetores densos. Podemos conectá-los diretamente ao codificador de um transformador como se fossem palavras embedded.

Encerramento
Se você ainda está comigo, criativa leitora, obrigada. Espero que tenha valido a pena. Esta é a conclusão de nossa jornada. Começamos com um objetivo de criar um conversor de fala para texto para nosso computador imaginário controlado por voz. No processo, partimos dos blocos de construção mais básicos, contagem e aritmética, e reconstruímos um transformador do zero. Minha esperança é que, da próxima vez que você ler um artigo sobre a mais recente conquista no processamento de linguagem natural, você possa assentir com satisfação, tendo um modelo mental bastante bom do que está acontecendo nos bastidores.

Técnicas Avançadas em Modelos de Linguagem: Uma Expansão
Além dos transformadores, o campo dos modelos de linguagem tem evoluído rapidamente, com diversas técnicas avançadas que impulsionam a capacidade das máquinas de entender e gerar linguagem natural. Vamos explorar algumas dessas técnicas, aprofundando as informações e fornecendo referências para artigos importantes em cada área.

Modelos de Linguagem Autorregressivos
Os modelos de linguagem autorregressivos são a espinha dorsal de muitos sistemas de geração de texto que vemos hoje. A ideia central é simples, mas poderosa: gerar texto sequencialmente, palavra por palavra (ou token por token), prevendo cada elemento com base no contexto das palavras que o precedem. Pense neles como escritores virtuais que, a cada passo, consultam o que já foi escrito para decidir o próximo passo na narrativa.

Um exemplo icônico dessa categoria são os modelos da família GPT (Generative Pre-trained Transformer) da OpenAI. Estes modelos, treinados em vastos conjuntos de dados textuais, demonstram uma capacidade notável de gerar textos coerentes, criativos e contextualmente relevantes, desde artigos e histórias até código de programação.

Artigos Importantes e Referências:

"Language Models are Few-Shot Learners" (GPT-3): Este artigo seminal introduz o GPT-3, demonstrando as capacidades impressionantes de modelos de linguagem em larga escala com poucos exemplos de treinamento.

Link para o artigo: https://arxiv.org/abs/2005.14165
"Improving Language Understanding by Generative Pre-Training" (GPT): O artigo original que apresenta o modelo GPT, estabelecendo as bases para a arquitetura e o treinamento de modelos autorregressivos transformadores.

Link para o artigo: https://openai.com/research/improving-language-understanding-by-generative-pre-training
Modelos de Linguagem Bidirecionais
Enquanto os modelos autorregressivos focam no contexto "para frente", os modelos de linguagem bidirecionais expandem a compreensão contextual ao considerar o contexto tanto "para frente" quanto "para trás" em uma sequência. O BERT (Bidirectional Encoder Representations from *Transformers*) é o exemplo mais proeminente desta categoria.

BERT é treinado para entender a linguagem em ambas as direções, o que o torna particularmente eficaz para tarefas de compreensão de linguagem natural (NLU), como classificação de texto, resposta a perguntas e reconhecimento de entidades nomeadas. Ao analisar uma palavra, BERT considera todas as palavras na frase, permitindo uma compreensão mais profunda e contextualizada.

Artigos Importantes e Referências:

"BERT: Pre-training of Deep Bidirectional *Transformers* for Language Understanding": O artigo que introduz o BERT, detalhando sua arquitetura, métodos de treinamento e resultados em diversas tarefas de NLU.
Link para o artigo: https://arxiv.org/abs/1810.04805
Modelos de Linguagem com Atenção Difusa (Soft Attention)
A atenção difusa, ou soft attention, é um mecanismo crucial em muitos modelos de linguagem modernos, especialmente em arquiteturas de transformadores. Em vez de simplesmente selecionar uma parte específica da entrada para focar (atenção hard), a atenção difusa permite que o modelo pondere a importância de diferentes partes da entrada de forma contínua.

Imagine traduzir uma frase: diferentes palavras na frase de origem podem ter diferentes graus de relevância para a palavra que está sendo gerada na tradução. A atenção difusa permite que o modelo atribua "pesos" de atenção a cada palavra de entrada, focando suavemente nas partes mais relevantes para a tarefa em questão.

Artigos Importantes e Referências:

"Neural Machine Translation by Jointly Learning to Align and Translate": Este artigo seminal introduz o mecanismo de atenção para tradução automática neural, revolucionando a área.

Link para o artigo: https://arxiv.org/abs/1409.0473
"Attention is All You Need": O artigo que apresenta a arquitetura Transformer, que depende fortemente do mecanismo de atenção para obter desempenho superior em diversas tarefas de NLP.

Link para o artigo: https://arxiv.org/abs/1706.03762
Modelos de Linguagem com Memória Externa
Para lidar com textos longos e manter a coerência ao longo de sequências extensas, os modelos de linguagem com memória externa incorporam um componente de memória explícito. Esta memória funciona como um "bloco de notas" para o modelo, permitindo armazenar e recuperar informações relevantes ao longo da geração do texto.

Em tarefas como geração de histórias longas ou diálogos complexos, a capacidade de lembrar detalhes e referências anteriores é crucial. A memória externa permite que o modelo consulte informações passadas, melhorando a consistência e a relevância do texto gerado.

Artigos Importantes e Referências:

"Neural Turing Machines": Este artigo introduz o conceito de máquinas de Turing neurais, que combinam redes neurais com mecanismos de memória externa, influenciando o desenvolvimento de modelos de linguagem com memória.

Link para o artigo: https://arxiv.org/abs/1410.5401
"End-To-End Memory Networks":  Apresenta redes de memória end-to-end, que aprendem a interagir com a memória de forma diferenciável, sendo aplicáveis a tarefas de resposta a perguntas e raciocínio.

Link para o artigo: https://arxiv.org/abs/1503.08895
Modelos de Linguagem com Reinforcement Learning
O Reinforcement Learning (RL) oferece uma abordagem alternativa para treinar modelos de linguagem, focando em otimizar o modelo para atingir um objetivo específico, em vez de apenas prever a próxima palavra. Em modelos de linguagem, o RL pode ser usado para refinar a qualidade do texto gerado, alinhando-o com critérios como coerência, relevância, ou estilo desejado.

Por exemplo, ao treinar um modelo para gerar resumos, a "recompensa" no RL pode ser uma métrica que avalia a qualidade do resumo em relação ao texto original. O modelo aprende a gerar resumos que maximizam essa recompensa, resultando em resumos mais informativos e concisos.

Artigos Importantes e Referências:

"Sequence to Sequence Learning with Neural Networks": Embora não diretamente sobre RL, este artigo estabeleceu a base para modelos sequenciais neurais, que são frequentemente combinados com RL para tarefas de geração de texto.

Link para o artigo: https://arxiv.org/abs/1409.3215
"Learning to Summarize from Human Feedback": Explora o uso de Reinforcement Learning com feedback humano para treinar modelos de sumarização de texto, mostrando como o RL pode melhorar a qualidade do resumo.

Link para o artigo: https://arxiv.org/abs/2006.06476
Modelos de Linguagem com Transferência de Estilo
A transferência de estilo é uma técnica que visa controlar o "tom de voz" de um modelo de linguagem, permitindo gerar texto em diferentes estilos, como formal, informal, poético, humorístico, etc. Isso abre portas para aplicações onde a adaptação ao estilo é crucial, como chatbots com personalidades distintas ou geração de conteúdo para diferentes públicos.

A transferência de estilo pode ser alcançada através de diversas abordagens, desde o uso de dados de treinamento estilisticamente diversos até a aplicação de técnicas de controle durante a geração do texto.

Artigos Importantes e Referências:

"Style Transfer from Non-Parallel Text by Cross-Lingual Training":  Um dos primeiros trabalhos a explorar a transferência de estilo textual, utilizando treinamento cross-lingual para alcançar a mudança de estilo.

Link para o artigo: https://arxiv.org/abs/1705.03192
"Controllable Text Generation":  Este artigo oferece uma visão geral das técnicas para geração de texto controlável, incluindo controle de estilo, e discute os desafios e direções futuras nesta área.

Link para o artigo: https://arxiv.org/abs/1905.12265
Modelos de Linguagem com Aprendizado Multimodal
O mundo não é feito apenas de texto, e os modelos de linguagem multimodais reconhecem isso. Eles integram informações de diversas modalidades de dados, como texto, imagens, áudio e vídeo, para construir uma compreensão mais rica e abrangente da linguagem e do mundo ao seu redor.

Por exemplo, um modelo multimodal pode ser treinado para gerar legendas para imagens, descrever o conteúdo de vídeos ou responder a perguntas que envolvem tanto texto quanto imagens. A fusão de diferentes modalidades permite que o modelo aprenda representações mais robustas e execute tarefas mais complexas.

Artigos Importantes e Referências:

"ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks": Introduz o ViLBERT, um modelo multimodal que aprende representações conjuntas de visão e linguagem, obtendo bons resultados em tarefas como resposta a perguntas visuais e legendagem de imagens.

Link para o artigo: https://arxiv.org/abs/1908.02265
"VisualBERT: A Simple and Performant Baseline for Vision and Language": Apresenta o VisualBERT, outro modelo multimodal que simplifica a arquitetura para tarefas de visão e linguagem, demonstrando desempenho competitivo com modelos mais complexos.

Link para o artigo: https://arxiv.org/abs/1908.03557
Modelos de Linguagem com Aprendizado Semi-Supervisionado
A disponibilidade de grandes quantidades de dados rotulados é frequentemente um gargalo no aprendizado de máquina. O aprendizado semi-supervisionado surge como uma solução inteligente, permitindo que os modelos aprendam com uma combinação de dados rotulados (onde temos a "resposta correta") e dados não rotulados (onde não temos rótulos).

Em modelos de linguagem, o aprendizado semi-supervisionado pode ser usado para aproveitar a vasta quantidade de texto não rotulado disponível na internet, complementando os dados rotulados mais escassos e melhorando o desempenho do modelo, especialmente em tarefas com poucos dados rotulados.

Artigos Importantes e Referências:

"Semi-Supervised Sequence Learning": Explora técnicas de aprendizado semi-supervisionado para tarefas de sequência a sequência, como tradução automática, mostrando como dados não rotulados podem ser usados para melhorar o desempenho.

Link para o artigo: https://arxiv.org/abs/1511.06732
"Noisy Student Training for Semi-Supervised Sequence Learning":  Apresenta uma abordagem de "aluno ruidoso" para aprendizado semi-supervisionado, onde um modelo "professor" rotula dados não rotulados para treinar um modelo "aluno", que pode ser repetido iterativamente para melhoria contínua.

Link para o artigo: https://arxiv.org/abs/1911.03907
Modelos de Linguagem com Aprendizado por Transferência
O aprendizado por transferência é um paradigma fundamental no aprendizado de máquina moderno. A ideia é reutilizar o conhecimento aprendido em uma tarefa (a tarefa "fonte") para melhorar o aprendizado em uma nova tarefa (a tarefa "alvo"). Em modelos de linguagem, isso é incrivelmente útil, pois permite que modelos pré-treinados em grandes corpora textuais sejam adaptados rapidamente para tarefas específicas com menos dados de treinamento.

Modelos como BERT e GPT são exemplos de modelos pré-treinados que podem ser "ajustados" (fine-tuned) para diversas tarefas de PNL, como classificação de sentimentos, resposta a perguntas e muito mais. O aprendizado por transferência economiza tempo e recursos de treinamento, além de melhorar o desempenho em tarefas com dados limitados.

Artigos Importantes e Referências:

"How transferable are features in deep neural networks?": Um estudo inicial que investiga a transferibilidade de características aprendidas em redes neurais profundas, mostrando que características de camadas iniciais são mais gerais e transferíveis.

Link para o artigo: https://papers.nips.cc/paper/2014/file/61e7588e2df15db26e50e839fef63fec-Paper.pdf
"Universal Language Model Fine-tuning for Text Classification":  Apresenta o ULMFiT, um método eficaz de fine-tuning para modelos de linguagem pré-treinados em tarefas de classificação de texto, demonstrando a praticidade do aprendizado por transferência em PNL.

Link para o artigo: https://arxiv.org/abs/1801.06146
Este texto expandido oferece uma visão mais aprofundada sobre as técnicas avançadas em modelos de linguagem, com referências a artigos importantes que podem ser explorados para um estudo mais detalhado. O campo continua a evoluir rapidamente, e estas técnicas representam apenas uma parte do vasto e fascinante panorama da pesquisa em modelos de linguagem.