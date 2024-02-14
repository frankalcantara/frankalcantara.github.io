---
layout: post
title: "A Fórmula da Atração: a Matemática do Eletromagnetismo"
author: Frank
categories:
  - Math
  - Electromagnetism
tags:
  - Math
  - Physics
  - Eletromagnetism
image: assets/images/eletromag1.jpg
description: Understand how mathematics underpins electromagnetism and its practical applications in an academic article aimed at science and engineering students.
slug: formula-of-attraction-mathematics-supporting-electromagnetism
keywords:
  - Vectorial Calculus
  - Eletromagnetism
  - Math
  - Poetry
  - Vectorial Algebra
rating: 5
---

Electromagnetism is the law, the ordering that cradles the universe. Like an ancient deity that governs the existence and movements of everything that exists. Two forces, electric and magnetic, in an endless dance, shape everything from a grain of dust to an ocean of stars. Even the very device you use to decipher these words owes its existence and functioning to Electromagnetism.

Imagem de [Asimina Nteliou](https://pixabay.com/users/asimina-1229333/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2773167) de [Pixabay](https://pixabay.com//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2773167)

> "Nesta longa vida eu aprendi que toda a nossa ciência se comparada com a realidade é primitiva e infantil. Ainda assim, **é a coisa mais preciosa que temos**." Albert Einstein

- [Vector Algebra](#vector-algebra)
  - [Vetores, os compassos de tudo que há e haverá](#vetores-os-compassos-de-tudo-que-há-e-haverá)
    - [Exercise 1](#exercise-1)
  - [Unit Vectors](#unit-vectors)
    - [Exercise 2](#exercise-2)
    - [Exercise 3](#exercise-3)
  - [Scalar Multiplication](#scalar-multiplication)
  - [Scalar Multiplication Properties](#scalar-multiplication-properties)
  - [Opposite Vector](#opposite-vector)
  - [Addition and Subtraction of Vectors](#addition-and-subtraction-of-vectors)
  - [Exercício 4](#exercício-4)
    - [Exercício 5](#exercício-5)
    - [Exercício 6](#exercício-6)
    - [Exercício 7](#exercício-7)
    - [Exercício 8](#exercício-8)
  - [Vetores Posição e Distância](#vetores-posição-e-distância)
    - [Exercício 9](#exercício-9)
    - [Exercício 10](#exercício-10)
  - [Produto Escalar](#produto-escalar)
    - [Exercício 11](#exercício-11)
    - [Exercício 12](#exercício-12)
    - [Exercício 13](#exercício-13)
  - [Produto Vetorial](#produto-vetorial)
    - [Exercício 14](#exercício-14)
  - [Produto Triplo Escalar](#produto-triplo-escalar)
- [PRECISA REESCREVER PARA INCLUIR O CONCEITO DA REGRA DA MÃO DIREITA](#precisa-reescrever-para-incluir-o-conceito-da-regra-da-mão-direita)
- [Usando a Álgebra Vetorial no Eletromagnetismo](#usando-a-álgebra-vetorial-no-eletromagnetismo)
  - [Lei de Coulomb](#lei-de-coulomb)
- [Cálculo Vetorial](#cálculo-vetorial)
  - [Campos Vetoriais](#campos-vetoriais)
  - [Gradiente](#gradiente)
    - [Significado do Gradiente](#significado-do-gradiente)
    - [Propriedades do Gradiente](#propriedades-do-gradiente)
  - [Divergência](#divergência)
    - [Fluxo e a Lei de Gauss](#fluxo-e-a-lei-de-gauss)
    - [Teorema da Divergência](#teorema-da-divergência)
    - [Propriedades da Divergência](#propriedades-da-divergência)
  - [Rotacional](#rotacional)

We will study invisible lines of force that intertwine, tangent, and interfere with each other, forming the fabric of the Cosmos and the flow of life, as real as the earth under our feet or the air we breathe, and like the latter, completely invisible.

The study of Electromagnetism will be a battle of its own, individual, hard. It is the hope to shed light on the unknown, to discover the rules that govern life and the universe, and then harness these rules to create, to progress, to survive. It is not for the faint of heart, nor for those who seek easy answers. It is for those who do not fear the unknown, for those who stand before the abyss of the unknown and say: _I will understand_.

The study of Electromagnetism is a challenge, a fight, a call. And, as in any fight, there will be losses, pains, but also joys, triumphs, and, at the end of it all, a different understanding of everything around you. This is a journey that began thousands of years ago and should continue for as long. Prepare yourself, your way of seeing the universe will change.

As the 19th century drew to a close, [James Clerk Maxwell](https://en.wikipedia.org/wiki/James_Clerk_Maxwell), orchestrated the dances of the electric and magnetic fields into a symphony of equations. Drawing on the canvas of the universe, Maxwell outlined the interaction of these forces with space and matter. His work, extraordinary in all aspects, stands out for its refined simplicity and lyrical beauty. A ballet of numbers, symbols, and equations that glides across the page, fluid and elegant like a river.

<div class="floatRight">

<img class="lazyimg" src="/assets/images/jcMaxwell.jpg" alt="Photograph of James Clerk Maxwell">

<legend class="legenda">Figure 1 - James Clerk Maxwell. Source: <a href="https://en.wikipedia.org/wiki/James_Clerk_Maxwell/" target="_blank">Wikipedia</a></legend>
</div>

But this beauty, this simplicity, is not accessible to all. It is a walled garden, reserved for those who have earned the right to enter through study and understanding. Without the appropriate knowledge, whether of the physics that underpins the universe or the mathematics that describes it, Maxwell's equations are like stone flowers: cold, unchanging, lifeless. With this understanding, however, they bloom in wonderful colors and shapes, alive and pulsating with meaning.

Here we embark on our journey! An exploration of this garden of stones and shadows, in search of the hidden beauty in the cold harshness of ignorance. In this book, our focus will be on understanding the mathematics that defines and explains the equations that structure the understanding of the universe. Our interest will begin in the abstract, in the pure, in the dance of numbers and symbols that make up these equations, and will gradually end in the analysis of the phenomena that mathematics explains. There is a vain hope that this strategy will create the cognitive structures that the kind reader will need to understand these phenomena and go beyond this modest introduction.

Consider this text as if you were releasing the ship of your knowledge from the ropes of the port of ignorance. This is the beginning of your journey on a sea of doubts. Intriguing, exciting, and provocative doubts. Almost a riddle. Do you like riddles? Need to overcome these challenges? Have the satisfaction of solving them? If so, embark on this journey in search of the most structuring knowledge of the Universe. Maybe you'll get there, maybe not.

Tears of disappointment will not find you at every port. That I can guarantee. Just as I can foresee all the fatigue, frustration, and pain resulting from the effort of learning. Learning is never easy. Even if our ship does not reach the desired destination. Each port of understanding will bring you the gift of knowledge in the end you will be a different person. They are rough seas, it will not be easy. But we will be together. I will be at the helm all the time, trying to find good winds and avoid rough seas. As [the poet](https://en.wikipedia.org/wiki/Fernando_Pessoa) would say:

> "...Everything is worth it If the soul is not small...". Fernando Pessoa.

## Vector Algebra

Area of mathematics involved with space, vectors, and their timeless dance, rhythmically dictated by intrinsic rules and characteristics of space. Vectors and Matrices, soldiers organized in rows and columns, each telling stories of variables and transformations. Divergences, gradients, and curl, majestic gestures in the dance of vector calculus. Everything as complex as life, as real as death, as honest as the sea, deep, merciless, and direct.

>The rough sea only respects a king. [Arnaud Rodrigues / Chico Anísio](https://www.letras.com/baiano-os-novos-caetanos/1272051/)

Space will be defined by vectors, filled with mystery and beauty. Vector analysis will be the navigator's compass, guiding them through the vast ocean of the unknown. Each day, each calculation, we will unveil a bit more of this infinity, map a bit more of this ocean of numbers, directions, senses, and values, understanding a little more about how the Universe dances to the sound of linear algebra and vector analysis.

There is a small difference between vector algebra and linear algebra. The latter has a rigid and formal structure that, whenever possible, I will ignore in the hope of simplifying comprehension. That's why I chose the field of vector algebra. The same field of study, with less formality and more application. This is a trade-off, exchanging the beauty of mathematical rigidity for simplicity in operations. The vector is the primitive element of vector algebra, and it is where we will begin.

### Vetores, os compassos de tudo que há e haverá

Vectors, silent beams of information, lead understanding beyond mere size. They are like compasses with a measure, pointing with determination and direction to unveil the secrets of magnitudes that require more than just magnitude. Vectors, mathematical abstractions we use to understand quantities that need direction and sense beyond pure magnitude. They seem to be the result of the brilliant mind of [Simon Stevin](https://en.wikipedia.org/wiki/Simon_Stevin), who, while studying mechanics and hydrostatics, proposed an empirical rule to resolve the problem of two or more forces applied at the same point through what we now know as the Parallelogram Rule, published in _De Beghinselen der Weeghconst_ (1586; loosely translated: Statics and Hydrostatics). We use vectors to overcome the limitations of scalar magnitudes, including in a single representation amplitude, direction, and sense.

Scalar quantities, those that can be measured like the mass of a fish, the time it takes for the sun to set, or the speed of a sailboat cutting the landscape in a line, I wish, straight. Each scalar quantity is a unique number, a quantity, a magnitude. A fact in itself carrying all the necessary knowledge for its understanding [^1].

They are the silent storytellers of the world, speaking of size, quantity, intensity. And, like good whiskey, their strength lies in simplicity. Yet, scalar quantities offer a measure of truth.

Vector quantities, on the other hand, are complex, diverse, and intriguing. Vectors are the abstractions to understand these quantities, warriors of direction and sense. They navigate the sea of mathematics with a clarity of purpose that goes beyond mere magnitude. They possess an arrow, a compass, indicating where to move. They surpass the value itself with a direction, a sense, an indication, an arrow. And so, we use arrows in respect to the ideas of [Gaspar Wessel](https://en.wikipedia.org/wiki/Caspar_Wessel) who in his work [_On the Analytical Representation of Direction_](https://web.archive.org/web/20210709185127/https://lru.praxis.dk/Lru/microsites/hvadermatematik/hem1download/Kap6_projekt6_2_Caspar_Wessels_afhandling_med_forskerbidrag.pdf) from 1778 suggested the use of _oriented lines_ to indicate the point where two lines intersect the same concept we use in vector sums. The arrow, our contemporary friend, emerges in the work of [Jean-Victor Poncelet](https://en.wikipedia.org/wiki/Jean-Victor_Poncelet), working on engineering problems and using the rules defined by Stevin and the concept of direction by Wessel, decided to use an arrow to indicate a force. And thus, the hands and mind of an engineer brought light to vectors.

The arrow, an extension of the vector itself, represents its orientation. It points the way to the truth, showing not just how much, but also where. It indicates its magnitude, how much, its essence. Thus vectors conceal intensity, direction, and sense in a single, fleeting, and intriguing entity.

Vector quantities are like the wind, whose direction and strength you feel, but whose essence you cannot grasp. They are like the river, whose flow and direction shape the landscape. They are essential for understanding the world in motion, the world of forces, speeds, and accelerations. They dance in the equations of electromagnetism, draw the patterns of physics, and guide sailors in the vastness of the unknown. In the sea of understanding, vector quantities are the compass and the wind, providing not just scale, but also orientation and sense to our quest for knowledge. How beautiful is the language of Machado de Assis, but, from time to time, we must resort to images. All this poetry can be summarized in the geometry of an arrow with origin and destination in a multidimensional space containing information on direction, sense, and intensity.

<div class="floatLeft">

<img class="lazyimg" src="/assets/images/vetor1.jpg" alt="Geometric representation of a vector, an arrow, going from one point to another.">

<legend class="legenda">Figure 2 - Geometric representation of a vector.</legend>
</div>

In this journey, we will not be limited by the coldness of geometry. We seek the grandeur of algebra. In algebra, vectors are represented by sum operations among other vectors.

I will try, I swear I will try, to limit the use of geometry to the minimum necessary for understanding the concepts related to the application of forces, and fields, which we will use to understand the universe of electromagnetism.

In modern physics, especially quantum physics, we use vectors as defined by [Dirac](https://en.wikipedia.org/wiki/Paul_Dirac) (1902-1984), known as Ket Vectors, or simply ket. Not here, at least not for now. Here, we will use the vector representation as defined by [Willard Gibbs](https://en.wikipedia.org/wiki/Josiah_Willard_Gibbs) (1839–1903) in the late 19th Century. Suitable for the classical study of Electromagnetism. The study of the forces that weave vector fields that embrace the very structure of the Universe. Invisible yet relentless.

Understanding these fields is a way to start understanding the universe. It is reading the story being written in the invisible lines of force. It is diving into the deep sea of the unknown, and emerging with new and precious knowledge. It is becoming a translator of the cosmic language, a reader of the marks left by forces in their fields. In short, it is the essence of science. And it is this science, this study of fields and the forces acting within them, that we will explore. For us, curious souls, fields will be functions capable of specifying values at any point in a given region of space [^1].

To lay the foundation stones of our knowledge, we will represent vectors using uppercase Latin letters $\, \vec{A}, \vec{B}, \vec{C}, ...$ marked with a small arrow. These vectors will be the building blocks of a vector space $\mathbf{V}$. As the reader can see, vector spaces will also be represented by uppercase Latin letters, this time in bold.

In this introductory text, the map of our journey, vector spaces will always be represented in three dimensions. The space we seek is ours, the space where we live, the way we perceive seas, mountains, plains, the sky, our universe.

It is not just any space, it is a specific space, limited to reality and limiting the operations we can perform with the elements of this space. Thus, our study will be based on a specific vector space. A vector space $\mathbf{V}$ that satisfies the following conditions:

1. the vector space $\mathbf{V}$ is closed with respect to addition. This means that for every pair of vectors $\vec{A}$ and $\vec{B}$ belonging to $\mathbf{V}$, there exists one, and only one, vector $\vec{C}$ that represents the sum of $\vec{A}$ and $\vec{B}$ and also belongs to the vector space $\mathbf{V}$, we say that:

    $$\exists \, \vec{A} \in \mathbf{V} \wedge \exists \vec{B} \in \mathbf{V} \therefore \exists (\, \vec{a}+\vec{B}=\vec{C}) \in \mathbf{V}$$

2. addition is associative:

   $$(\, \vec{A}+\vec{B})+\vec{C} = \, \vec{a}+(\vec{B}+\vec{C})$$

3. there is a zero vector: adding this zero vector to any vector $\, \vec{A}$ results in the vector $\, \vec{A}$ itself, unchanged, immutable. Such that:

   $$\forall \, \vec{A} \in \mathbf{V} \space \space \exists \wedge \vec{0} \in \space \mathbf{V} \space \therefore \space \vec{0}+\, \vec{A}=\, \vec{A}$$

4. there is a negative vector $-\, \vec{A}$ such that the sum of a vector with its negative vector results in the zero vector. Such that:

   $$\exists -\, \vec{A} \in \mathbf{V} \space \space \vert \space \space -\, \vec{A}+\, \vec{A}=\vec{0}$$

5. the vector space $\mathbf{V}$ is closed with respect to multiplication by a scalar, a value without direction or sense, so that for every element $c$ of the set of complex numbers $\mathbb{C}$ multiplied by a vector $\, \vec{a}$ from the vector space $\mathbf{V}$, there exists one, and only one vector $c\, \vec{a}$ that also belongs to the vector space $\mathbf{V}$. Such that:

   $$\exists \space c \in \mathbb{C} \space \space \wedge \space \space \exists \space \, \vec{A} \in \mathbf{V} \space \space \therefore \space \space \exists \space c\, \vec{A} \in \mathbf{V}$$

6. There is a neutral scalar $1$: such that the multiplication of any vector $\vec{A}$ by $1$ results in $\, \vec{A}$. That is:

   $$\exists \space 1 \in \mathbb{R} \space \space \wedge \space \space \exists \space \, \vec{A} \in \mathbf{V} \space \space \vert \space \space 1\, \vec{A} = \, \vec{A}$$

Attention must be paid to the hierarchy governing the world of sets. The set of real numbers $\mathbb{R}$ is a subset of the set of imaginary numbers $\mathbb{C}=\{a+bi \space \space a.b \in \mathbb{R}\}$. This relationship of belonging determines that the set $\mathbb{R}$, the set of real numbers, when seen more broadly, concisely represents all imaginary numbers whose imaginary part is equal to zero. Using the language of mathematics, we say:

$$\mathbb{R}=\{a+bi \space \space \vert \space \space a.b \in \mathbb{R} \wedge b=0\}$$

The algebraic representation of vectors defined by [Willard Gibbs](https://en.wikipedia.org/wiki/Josiah_Willard_Gibbs) (1839–1903), which we will use in this document, indicates that a vector in any vector space $\mathbf{V}$ is, simply, the result of operations performed between the vectors that define the components of this vector space.

We already know that our space $\mathbf{V}$ will be formed in three dimensions, so we need to choose a set of coordinates that define the points of this space and use these points to determine the vector components that we will use to specify all vectors of the space $\mathbf{V}$.

[Maxwell](https://en.wikipedia.org/wiki/James_Clerk_Maxwell), following in the steps of [Newton](https://en.wikipedia.org/wiki/Isaac_Newton), also stood on the shoulders of giants. And here, in our journey, we encounter one of these giants. In the mid-17th century, [René Descartes](https://plato.stanford.edu/entries/descartes/) created a coordinate system defining the space we know. So precise, simple, and efficient that it prevailed against time and still bears the Latin name of its creator: **Cartesian Coordinate System**.

Vectors are arrows that represent forces, a metaphor, a mathematical abstraction that allows the understanding of the universe through the analysis of the forces that compose, define, and move it. We define a vector simply by observing its origin and destination points, marking these points in space and drawing an arrow connecting these two points. And this we leave in the domain of geometry whenever we can. In the algebra of the real world, the one we are studying, vectors will be defined by subtraction. The vector will be defined by the coordinates of the destination point and the origin point. It becomes clear when we use the Cartesian Coordinate System.

#### Exercise 1

On a hot afternoon in a seaside bar, an old fisherman was talking to a young apprentice about vectors. "They are like the wind, they have direction, sense, and intensity," said the fisherman. "Imagine two points in the sea, and we want to know the direction and strength of the wind between them." He drew on the ground with a piece of charcoal the points: A(1,2,3) and B(-1,-2,3). "Now," he asked, "how do we determine the vector between these two points?"

In the Cartesian Coordinate System, we limit the space with three axes, perpendicular and orthogonal, and by the values of the coordinates $(x,y.z)$ placed on these axes. From the point of view of Vector Algebra, for each of these axes, we will have a unit length vector. These vectors, which we call unit vectors and identify by $(\, \vec{a}_x, \, \vec{a}_y, \, \vec{a}_z)$ respectively, are called unit vectors because they have magnitude $1$ and are oriented according to the Cartesian axes $(x,y,z)$.

Remember: **the magnitude of a vector is its length. Unit vectors have a length of $1$**.

The charm of mathematics presents itself when we say that all vectors of the vector space $\mathbf{V}$ can be represented by sums of the unit vectors $(\, \vec{a}_x, \, \vec{a}_y, \, \vec{a}_z)$ as long as these vectors are independently multiplied by scalar factors. This implies, even if it is not clear now, that **any vector in space will be the product of a unit vector by a scalar**. To clarify, we need to understand the unit vectors.

### Unit Vectors

Any vector $\vec{B}$ has magnitude, direction, and sense. The magnitude, also called intensity, modulus, or length, will be represented by $\vert \vec{B} \vert$. We define a unit vector $\, \vec{a}$ in the direction $\vec{B}$ as $\, \vec{a}_B$ such that:

$$ \, \vec{a}_B=\frac{\vec{B}}{|\vec{B}|} $$

A unit vector $\, \vec{a}_B$ is a vector that has the same direction and sense as $\vec{B}$ with a magnitude of $1$. Therefore, the modulus, or magnitude, or length of $\, \vec{a}_b$ will be represented by:

$$\vert \, \vec{a}_B \vert=1$$

Now that we understand unit vectors, we can comprehend the rules that sustain Vector Algebra and make it possible for all the geometric concepts that underlay the existence of vectors to be algebraically represented, without lines or angles, in a space, as long as this space is algebraically defined in a coordinate system. Here, we will use three-dimensional coordinate systems.

In a three-dimensional orthogonal coordinate system, we can express any vector in the form of the sum of its orthogonal unit components. Any vector, regardless of its direction, sense, or magnitude, can be represented by the sum of the unit vectors that represent the directions, axes, and coordinates of the chosen coordinate system. Each factor of this sum will be called a _vector component_, or simply component. There will be one component for each dimension of the coordinate system, and these components are specific to the coordinate system chosen to represent the space we will call $\mathbf{V}$.

As we are novices, we sail during the day, in known seas, keeping the land in sight. In this case, we will start with the Cartesian Coordinate System. A known, safe, and easy-to-represent coordinate system. It won't be difficult to visualize a vector space defined in this system since it's the space we live in. Your living room has a width $x$, a length $y$, and a height $z$. In the Cartesian Coordinate System, the representation of any vector $\vec{B}$ according to its orthogonal unit components will be given by:

$$\vec{B}=b_x\, \vec{a}_x+b_y\, \vec{a}_y+b_z\, \vec{a}_z$$

In this representation, $b_x$, $b_y$, $b_z$ represent the scalar factors that we must use to multiply the unit vectors $\, \vec{a}_x$, $\, \vec{a}_y$, $\, \vec{a}_z$ so that the sum of these vectors represents the vector $B$ in the space $\Bbb{R}^3$.

Here we call $b_x$, $b_y$, $b_z$ the vector components in the directions $x$, $y$, $z$, or the projections of $\vec{B}$ on the axes $x$, $y$, $z$. The proof of equivalence between the components and the projections on the axes belongs to the domain of geometry that we left at the port when we started this journey.

The simplicity of the Cartesian Coordinate System is also its curse. Studying Electromagnetism, we will face many problems in which the use of this system will make the mathematics unnecessarily torturous. In this case, we can resort to any other coordinate system, with the only condition of having three orthogonal dimensions. For example, we could define our vector $\vec{B}$ as:

$$\vec{B}=b_x\, \vec{a}_x+b_y\, \vec{a}_y+b_z\, \vec{a}_z$$

$$\vec{B}=b_r\, \vec{a}_r+b_\phi \, \vec{a}_\phi+b_z\, \vec{a}_z$$

$$\vec{B}=b_r\, \vec{a}_r+b_\phi \, \vec{a}_\phi+b_\theta \, \vec{a}_\theta$$

Respectively for the _Cartesian, Cylindrical, and Spherical Coordinate Systems_.

Different coordinate systems for the same space are like different nautical maps for the same ocean. On each map, north is still north, and a constellation will still guide you to port. The vector $\vec{B}$ maintains its magnitude, direction, and sense, no matter which nautical chart you unroll on the table. And when it is necessary to transition between these systems, we will do so with the precision of an experienced fisherman diving into the blue depths to retrieve a precious harpoon. Have no doubt, the essential will remain constant; only the medium changes. And, when necessary, we will study these systems to understand how different maps show the same ocean.

Mathematics, like the sea, holds its own surprises. Sometimes, after defining the coordinate system as an old sailor chooses his route, the vectors strip down to their simplest essences, their vector components, leaving behind the unit vectors like a ship abandons its ballast. In the Cartesian Coordinate System, the vector $\vec{B} = 3\, \vec{a}_x + \, \vec{a}_y - \, \vec{a}_z$ transforms and can be represented just by its coordinates $\vec{B} = (3, 1, -1)$, like a ship that hoisted its sails, ready for the journey, free from unnecessary weight. The substance remains, while the form adapts to the challenge of the moment. During the arduous task of solving your problems, you will have to choose how you will represent your vectors. I, fickle as I am, will sometimes write $\vec{B} = 3\, \vec{a}_x + \, \vec{a}_y - \, \vec{a}_z$ and other times write $\vec{B} = (3, 1, -1)$. It will be up to the patient reader the task of interpretation, the extract of attention and learning.

If we have a vector $\vec{B} = b_x\, \vec{a}_x + b_y\, \vec{a}_y + b_z\, \vec{a}_z$ its magnitude will be given by:

$$ \vert \vec{B} \vert=\sqrt{ {b_x}^2 + {b_y}^2 + {b_z}^2}$$

At first, it may escape the perception of the kind reader, but it is a fact that in this way we can find the unit vector ${\, \vec{a}_B}$, which we will read as the unit vector a in the direction of vector $\vec{B}$ by:

$$\, \vec{a}_B=\frac{ \vec{B} }{ \vert \vec{B} \vert }= \frac{b_x\, \vec{a}_x+b_y\, \vec{a}_y+b_z\, \vec{a}_z}{ \sqrt{b_x^2+b_y^2+b_z^2} }$$

This equation should be read as: **the unit vector of a given vector will be the vector itself divided by its magnitude**. Perhaps all this mathematical rigidity will vanish before your beautiful and tired eyes if we resort to an example.

<p class="exp">
<b>Example 1:</b><br>
Calculate the unit vector $\, \vec{a}_A$ of the vector $\, \vec{a}=\, \vec{a}_x-3\, \vec{a}_y+2\, \vec{a}_z$. <br><br>
<b>Solution:</b> Starting from the definition of a unit vector.

\[\, \vec{a}_A=\frac{\, \vec{a}_x\, \vec{a}_x+\, \vec{a}_y\, \vec{a}_y+\, \vec{a}_z\, \vec{a}_z}{\sqrt{\, \vec{a}_x^2+\, \vec{a}_y^2+\, \vec{a}_z^2} }\]

Substituting the given values:

\[\, \vec{a}_A=\frac{\, \vec{a}_x-3\, \vec{a}_y+2\, \vec{a}_z}{\sqrt{1^2+(-3)^2+2^2} }=\frac{\, \vec{a}_x-3\, \vec{a}_y+2\, \vec{a}_z}{3.7416}\]

\[\, \vec{a}_A=0.2672\, \vec{a}_x-0.8018\, \vec{a}_y+0.5345\, \vec{a}_z\]

</p>

<p class="exp">
<b>Example 2:</b><br>
Given points $A(2, 1, 3)$ and $B(4, -2, 5)$, find the vector $\vec{V}_{AB}$ and the unit vector $\vec{v}_{AB}$.
<br><br>
<b>Solution:</b>
To find the vector $\vec{V}_{AB}$, we use the equation:

\[
\vec{V}_{AB} = B - A = (4 \, \vec{a}_x - 2 \, \vec{a}_y + 5 \, \vec{a}_z) - (2 \, \vec{a}_x + 1 \, \vec{a}_y + 3 \, \vec{a}_z)
\]

Simplifying, we get:
\[
\vec{V}_{AB} = 2 \, \vec{a}_x - 3 \, \vec{a}_y + 2 \, \vec{a}_z
\]

To find the unit vector $\vec{v}_{AB}$:
\[
\vec{v}_{AB} = \frac{\vec{V}_{AB} }{\vert \vec{V}_{AB} \vert } = \frac{2 \, \vec{a}_x - 3 \, \vec{a}_y + 2 \, \vec{a}_z}{\sqrt{2^2 + (-3)^2 + 2^2} }
\]
After calculating the modulus:
\[
|\vec{V}_{AB}| = \sqrt{4 + 9 + 4} = \sqrt{17} \approx 4.1231
\]
Therefore, the unit vector $\vec{v}_{AB}$ will be given by:
\[
\vec{v}_{AB} = \frac{2 \, \vec{a}_x - 3 \, \vec{a}_y + 2 \, \vec{k}_z}{4.1231} \approx 0.4845 \, \vec{a}_x - 0.7267 \, \vec{a}_y + 0.4845\vec{a}_z
\]

</p>

Viewed through attentive retinas, mathematics is simple and often beautiful.

#### Exercise 2

You are the captain of a small fishing boat, lost at sea. Your compass, printed on a Cartesian plane, shows the direction to a safe harbor as a vector $\, \vec{a} = (4, 3, -1)$. This vector contains the direction and force of the winds and currents you must face. Your task is to simplify this information into a unit vector that points precisely to the harbor. Remember, a unit vector has a magnitude of $1$ and points in the same direction and sense as the original vector. Use your skills in vector algebra to find this unit vector and point your boat home.

#### Exercise 3

On an ancient map of a solitary navigator, distances were indicated only by units, without a specific definition of their measure, as if they were steps or handspans. In those times, precision wasn't as demanding, and navigators often relied on their instincts and observational skills. On this peculiar map, the navigator noted:

1. A route that starts at his point of departure, marked as the origin, and goes to a point of interest $A = (-3, 4, 5)$.
2. A unit vector $b$ that, also starting from the origin, points in the direction of a second point of interest, $B$, represented by $\vec{b} = \frac{(-2, 1, 3)}{2}$.
3. He also made a note that the distance between the two points of interest, $A$ and $B$, was 12 units. Perhaps this was the distance he needed to travel in one day to reach point $B$ before nightfall. Perhaps it was just a dream, a destination never traversed. We don't know, but it might be possible to determine the exact coordinates of point $B$ on the map.

Considering the available information, what would be the exact location of point $B$ on the map?

### Scalar Multiplication

A scalar is a number, a value, cold, simple, and direct. The information contained in a scalar does not need direction, sense, or any other information. The mass of your ship is a scalar value, the speed at which it sails the seas is a vector value.

Clearly, all real numbers $(\mathbb{R})$, integers $(\mathbb{Z})$, and natural numbers $(\mathbb{N})$ are scalars. Complex numbers $(\mathbb{C})$ are also scalars, but they require a bit more attention.

**Complex numbers**, $\mathbb{C}$, contain information that may be associated with direction and sense, but they are not vectors. They are like fish in a lake. The real part is like the distance the fish swims east or west. The imaginary part is how much it swims north or south. They can move in two directions, but they are not like wind or a river, which have a clear direction and sense. Complex numbers, they are more like the fish - swimming around, unconcerned with direction. **They are scalars, not vectors**.

Just as a fisherman marks the position of a fish by how far it is from the shore and at what angle, we can do the same with complex numbers. We call magnitude the distance to the origin and angle is the direction pointing to them. Still, do not confuse this with direction and sense of a vector in physics. It's a comparison, nothing more.

It's important to understand that complex numbers, $\mathbb{C}$, have a concept related to magnitude and phase, angle in polar representation, where a complex number $c$ can be represented as $r*e^{i\theta}$, where $r$ is the magnitude (or modulus) of the complex number, and $\theta$ is the phase (or argument), which can be thought of as the direction of the complex number in the complex plane. But again, the concept of direction used here is not the same as the concept of direction when referring to vectors. It is just a mathematical analogy.

Yes! Mathematics has analogies.

We will return to complex numbers when convenient to the understanding of electromagnetic phenomena. Let's just keep in our toolbox the notion that a number, whether complex or not, is a scalar. A piece of information of fundamental value for understanding operations with vectors.

The multiplication of a vector $\vec{B}$ by a scalar implies multiplying each of the components $b$ of this vector by this scalar.

The scalars we will use in this journey will be elements of the set of real numbers $\Bbb{R}$. Not forgetting that, as we saw before, the elements of the set of real numbers $\Bbb{R}$ are a subset of the set of complex numbers $\Bbb{C}$, the same definition we used when we explained the rules of formation of the vector space $\mathbf{V}$ when defining the universe in which we are navigating.

The multiplication of each component by a scalar is very simple and almost does not require an example. Almost.

<p class="exp">
<b>Example 3:</b> <br>
Consider the vector $\, \vec{a}=2\, \vec{a}_x+4\, \vec{a}_y-\, \vec{a}_z$ and calculate $3.3\, \vec{a}$ and $\, \vec{a}/2$: <br><br>
<b>Solution:</b>

\[3.3\, \vec{a}=(3.3)(2)\, \vec{a}_x+(3.3)(4)\, \vec{a}_y+(3.3)(-1)\, \vec{a}_z\]

\[3.3\, \vec{a}=6.6\, \vec{a}_x+13.2\, \vec{a}_y-3.3\, \vec{a}_z\]

\[\frac{ \, \vec{a} }{2}=(\frac{1}{2})(2)\, \vec{a}_x+(\frac{1}{2})(4)\, \vec{a}_y+(\frac{1}{2})(-1)\, \vec{a}_z\]

\[\frac{\, \vec{a} }{2}=\, \vec{a}_x+2\, \vec{a}_y-\frac{1}{2}\, \vec{a}_z\]

</p>

### Scalar Multiplication Properties

Scalar multiplication is commutative, associative, distributive, and closed with respect to zero and the neutral element. If we have scalars $m$ and $n$ and vectors $\, \vec{a}$ and $\vec{B}$, the properties of scalar multiplication are given by:

1. **Commutativity:** The order of the factors does not affect the product. Therefore, if you multiply a vector by a scalar, you will get the same result regardless of the order. That is, $m(\, \vec{a}) = (\, \vec{a})m$.

2. **Associativity:** The way factors are grouped does not affect the product. Therefore, if you multiply a vector by a product of scalars, you will get the same result regardless of how the factors are grouped. That is, $(mn)\, \vec{a} = m(n\, \vec{a})$.

3. **Distributivity:** Scalar multiplication is distributive with respect to the addition of vectors and scalars. Therefore, if you multiply the sum of two vectors by a scalar, the result will be the same as if you multiplied each vector by the scalar and added the results. That is, $m(\, \vec{a} + \vec{B})=m\, \vec{a} + m\vec{B}$. Similarly, if you multiply a vector by the sum of two scalars, the result will be the same as if you multiplied the vector by each scalar and added the results. That is, $(m + n)\, \vec{a} = m\, \vec{a} + n\, \vec{a}$.

4. **Closure with respect to zero and the neutral element:** Multiplying any vector by zero results in the zero vector. That is, $0\, \vec{a} = 0$. And multiplying any vector by $1$ (the neutral element of scalar multiplication) results in the same vector. That is, $1\, \vec{a} = \, \vec{a}$. In summary, we have:

    $$m\, \vec{a}=\, \vec{a}m$$

    $$m(n\, \vec{a}) = (mn)\, \vec{a}$$

    $$m(\, \vec{a}+\vec{B}) = m\, \vec{a}+m\vec{B}$$

    $$(\, \vec{a}+\vec{B})n = n\, \vec{a}+n\vec{B}$$

    $$1\, \vec{a}=\, \vec{a}$$

    $$0\, \vec{a}=0$$

### Opposite Vector

Multiplying a vector by the scalar $-1$ is special. We call the **opposite vector** to the vector $\, \vec{a}$ the vector that has the same intensity, the same direction, and the opposite sense to the sense of $\, \vec{a}$. An Opposite Vector is the result of multiplying a vector by the scalar $-1$. Therefore:

$$-1\, \vec{a} = -\, \vec{a}$$

It must be opposite. It opposes the magnitude that the vector represents. Vectors cannot be negative [^2]. There are no negative vectors just as there are no negative forces. Therefore, they must be opposite, opposing a direction in one sense.

A vector is a collection of information, a direction, a sense, and a magnitude. A tuple with three pieces of information, none of them can be negative. On the other hand, we know that forces can be pulls or pushes. If they are equal at a certain point, there is no effect. How to represent something that has the same intensity, the same direction, and the opposite sense? We use a negative sign. Without the opposite vector, arithmetic between vectors would be very complex, or impossible.

### Addition and Subtraction of Vectors

Look at the birds in the sky. Vectors are like the trail of a bird in the sky, showing not only how far it flew but also the direction it chose. They represent forces, those invisible winds that move the world, which are also like this. They have amplitude and direction; forces are vectors in the universe of Linear Algebra.

Like the birds in the sky, vectors can also join or separate. Addition, subtraction, are part of their flight. Some may find it useful to imagine this, resorting to geometry, like a parallelogram, a shape with parallel sides that shows how one vector adds to another.

I will not prioritize a journey through the world of shapes and lines, not here, not now. Even so, the kind reader needs to remember that geometry, silent and motionless, will always be there, underneath everything, the skeleton of the invisible that gives the physical shape of our universe. Using a bit of geometry, the addition of vectors can be easily visualized in a two-dimensional space, a plane. In this case, we can translate the involved vectors to create a parallelogram. The major diagonal of this parallelogram will represent the sum of the vectors.

<div class="floatRight">

<img class="lazyimg" src="/assets/images/SomaVetores.jpeg" alt="Parallelogram formed by the original vectors, their translations in space, and the representation of the sum vector.">

<legend class="legenda">Figure 3 - Vector addition using the parallelogram rule.</legend>
</div>

In Figure 3, the resultant vector $\vec{R}$, resulting from $\vec{P}+\vec{Q}$, can be deduced by applying trigonometry to the triangles formed by the translation of the vectors. We translate a copy of $\vec{Q}$ to point $A$ and translate a copy of $\vec{P}$ to point $D$, forming a parallelogram, $OABD$.

To determine the vector $\vec{R}$, we create a line perfectly overlaid on the vector $\vec{P}$ that extends beyond its length, and a straight segment, perfectly transverse to the extension of vector $\vec{P}$ passing through point $B$. We will call this perpendicular line segment $\overline{CB}$. At this point, besides the parallelogram that gives the rule its name, we form a right triangle, $OCB$.

Considering the line segments $\overline{OC}$, $\overline{OB}$, and $\overline{BC}$ that form the triangle $OCB$ and the Pythagorean Theorem, we have:

$$\overline{OB}^2 =\overline{OC}^2 + \overline{BC}^2 $$

Which can be written in a more convenient form if we divide $\overline{OC}$ into two line segments:

$$\overline{OB}^2 =(\overline{OA}+\overline{AC})^2 + \overline{BC}^2 \quad (\text{i})$$

Observing triangle $OCB$, and applying a bit of trigonometry, we can say:

$$\cos \theta = \frac{\overline{AC} }{\overline{BC} } \space\space \therefore \space\space \overline{AC} = (\cos \theta) (\overline{BC})$$

Since $\overline{AB} = \overline{OD} = \vert \vec{Q} \vert$, we have:

$$\overline{AC} = (\cos \theta) (\vert \vec{Q} \vert)$$

This is not the only triangle we have. We can also work with triangle $ABC$. In this case:

$$\cos \theta = \frac{\overline{BC} }{\overline{AB} } \space\space \therefore \space\space \overline{BC} = (\cos \theta) (\overline{AB})$$

The kind reader needs to look at this sum with care and attention. Observe that when we find the main diagonal, we also find an area. The area of the parallelogram.

Irritable mathematics forces us to say that the vector space $\mathbf{V}$ is closed with respect to the addition of vectors. This is a direct way of saying that the sum of two vectors in space $\mathbf{V}$ results in a vector of this same space. Closure is a concept from algebra, and it determines which binary operations applied to the elements of a set result in elements of this same set.

Limited as we are by Linear Algebra, we will see that the addition of vectors in a given vector space will be done component by component. If we consider vectors $\vec{A}$ and $\vec{B}$, we can

$$\vec{C}= \vec{A}+\vec{B}=(\, \vec{a}_x \, \vec{a}_x+\, \vec{a}_y \, \vec{a}_y+\, \vec{a}_z \, \vec{a}_z)+(B_x \, \vec{a}_x+B_y \, \vec{a}_y+B_z \, \vec{a}_z)$$

$$\vec{C}=\vec{A}+\vec{B}=(\, \vec{a}_x+B_x)\, \vec{a}_x+(\, \vec{a}_y+B_y)\, \vec{a}_y+(\, \vec{a}_y+B_y)\, \vec{a}_z$$

<p class="exp">
<b>Exemplo 4:</b><br>
Se \vec{A}=5\vec{a}_x-3\, \vec{a}_y+\, \vec{a}_z$ e $\vec{B}=\, \vec{a}_x+4\, \vec{a}_y-7\, \vec{a}_z$. Calcule $\vec{C}=vec{A}+\vec{B}$.<br><br>
<b>Solução</b>

\[\vec{C}=\vec{A}+\vec{B}=(5\, \vec{a}_x-3\, \vec{a}_y+\, \vec{a}_z)+(1\, \vec{a}_x+4\, \vec{a}_y-7\, \vec{a}_z)\]

\[\vec{C}=\vec{A}+\vec{B}=(5+1)\, \vec{a}_x+(-3+4)\, \vec{a}_y+(1-7)\, \vec{a}_z \]

\[\vec{C}= 6\, \vec{a}_x+\, \vec{a}_y-6\, \vec{a}_z\]

</p>

<p class="exp">
<b>Exemplo 5:</b><br>
Dado o vetor $\vec{A} = 4 \, \vec{a}_x + 6 \, \vec{a}_y + 3 \, \vec{a}_z$ e o vetor $\vec{B} = 3 \, \vec{a}_x - 2\vec{a}_y + 8 \, \vec{a_z}$, a projeção do vetor soma $\vec{C}=\vec{A}+\vec{B}$ sobre o eixo $y$ será:
.<br><br>
<b>Solução</b>

Para descobrir essa projeção, precisamos efetuar a soma $\vec{C}=\vec{A}+\vec{B}$:
\[
\vec{C}=\vec{A}+\vec{B} = (4 + 3) \, \vec{a}_x + (6 - 2) \, \vec{a}_y + (3 + 8) \, \vec{a}_z = 7 \, \vec{a}_x + 4 \, \vec{a}_y + 11 \, \vec{a}_z
\]
Lembrando que <b>os componentes do vetor são, na verdade, a projeção do vetor em cada um dos eixos do sistema de coordenadas</b>, a projeção sobre o eixo $y$ será: $4$.
</p>

Recorrendo ao auxílio da aritmética dos números escalares, podemos dizer que: a subtração entre dois vetores também será uma soma. Desta feita, uma soma entre um vetor e o vetor oposto de outro vetor Assim:

$$\vec{C}=\vec{A}-\vec{B}=\vec{A}+(-\vec{B})=\vec{A}+(-1\vec{B})$$

Talvez um exemplo ajude a amável leitora a perceber que, vetorialmente, até quando subtraímos estamos somando.

<p class="exp">
<b>Exemplo 6:</b><br> 
Considere $\vec{A}=\vec{a}_x-3\, \vec{a}_y+\, \vec{a}_z$ e $\vec{B}=1\, \vec{a}_x+4\, \vec{a}_y-7\, \vec{a}_z$ e calcule $\vec{C}=\, \vec{a}-\vec{B}$. <br><br>
<b>Solução:</b>

\[\vec{C}=\vec{A}-\vec{B}=(5\, \vec{a}_x-3\, \vec{a}_y+\, \vec{a}_z)+(-1(1\, \vec{a}_x+4\, \vec{a}_y-7\, \vec{a}_z))\]

\[\vec{C}=\vec{A}-\vec{B}=(5\, \vec{a}_x-3\, \vec{a}_y+\, \vec{a}_z)+(-1\, \vec{a}_x-4\, \vec{a}_y+7\, \vec{a}_z)\]

\[\vec{C}=\vec{A}-\vec{B}=4\, \vec{a}_x-7\, \vec{a}_y+8\, \vec{a}_z\]

</p>

A consistência ressalta a beleza da matemática. As operações de adição e subtração de vetores obedecem a um conjunto de  propriedades matemáticas que garantem a consistência destas operações. Para tanto, considere os vetores $\vec{A}$, $\vec{B}$ e $\vec{B}$, e o escalar $m$:

1. **comutatividade da adição de vetores:** a ordem dos vetores na adição não afeta o resultado final. Portanto, $\vec{A} + \vec{B} = \vec{B} + \, \vec{a}$. A subtração, entretanto, não é comutativa, ou seja, $\vec{A} - \vec{B} ≠ \vec{B} - \vec{A}$. A comutatividade é como uma dança onde a ordem dos parceiros não importa. Neste caso, subtrair não é como dançar e a ordem importa.

2. **associatividade da adição de vetores:** a forma como os vetores são agrupados na adição não afeta o resultado final. Assim, $(\vec{A} + \vec{B}) + \vec{C} = \vec{A} + (\vec{B} + \vec{C})$. A associatividade é como um grupo de amigos que se reúne. Não importa a ordem de chegada o resultado é uma festa. A subtração, entretanto, não é associativa, ou seja, $(\vec{A} - \vec{B}) - \vec{C} ≠ \vec{A} - (\vec{B} - \vec{C})$.

3. **Distributividade da multiplicação por escalar em relação à adição de vetores:** Se você multiplicar a soma de dois vetores por um escalar, o resultado será o mesmo que se você multiplicar cada vetor pelo escalar e somar os resultados. Isto é, $m(\vec{A} + \vec{B}) = m\vec{A} + m\vec{B}$.

Essas propriedades são fundamentais para a manipulação de vetores em muitas áreas da física e da matemática e podem ser resumidas por:

$$\vec{A}+\vec{B}=\vec{B}+\vec{A}$$

$$\vec{A}+(\vec{B}+\vec{C})=(\vec{A}+\vec{B})+\vec{C}$$

$$m(\vec{A}+\vec{B})=m\vec{A}+m\vec{C}$$

**Importante**: a subtração não é comutativa nem associativa. Logo:

$$\vec{A} - \vec{B} ≠ \vec{B} - \vec{A}$$

$$(\vec{A} - \vec{B}) - \vec{C} ≠ \vec{A} - (\vec{B} - \vec{C})$$

### Exercício 4

Alice é uma engenheira trabalhando no projeto de construção de uma ponte. As forças aplicadas sobre um pilar foram simplificadas até que serem reduzidas a dois vetores: $\vec{F}_1 = 4\, \vec{a}_x + 3\, \vec{a}_y$ e $\vec{F}_2 = -1\, \vec{a}_x + 2\, \vec{a}_y$ a força aplicada ao pilar será o resultado da subtração entre os vetores. Alice precisa saber qual será a força resultante após aplicar uma correção de segurança ao vetor  $\vec{F}_2$ multiplicando-o por $2$. O trabalho de Alice é definir as características físicas deste pilar, o seu é ajudar Alice com estes cálculos.

#### Exercício 5

Larissa é uma física estudando o movimento de uma partícula em um campo elétrico. Ela reduziu o problema a dois vetores representando as velocidades da partícula em um momento específico:
$\vec{V}_1 = 6\, \vec{a}_x - 4\, \vec{a}_y + 2\, \vec{a}_z$ e $\vec{V}_2 = 12\, \vec{a}_x + 8\, \vec{a}_y - 4\, \vec{a}_z$. Larissa precisa qual será a velocidade média da partícula se ele considerar que $\vec{V}_2$ deve ser dividido por $2$ graças ao efeito de uma força estranha ao sistema agindo sobre uma das partículas. Para ajudar Larissa ajude-a a determinar a velocidade média, sabendo que esta será dada pela soma destes vetores após a correção dos efeitos da força estranha ao sistema.

#### Exercício 6

Marcela é uma física experimental realizando um experimento em um laboratório de pesquisas em um projeto para estudar o movimento de partículas subatômicas. As velocidades das partículas $A$ e $B$ são representadas pelos vetores $\vec{v}_A$ e $\vec{v}_B$, definidos por:

$$ \vec{v}_A = -10\, \vec{a}_x + 4\, \vec{a}_y - 8\, \vec{a}_z \, \text{m/s} $$

$$ \vec{v}_B = 8\, \vec{a}_x + 7\, \vec{a}_y - 2\, \vec{a}_z \, \text{m/s} $$

Marcela precisa calcular a velocidade resultante $\vec{v}_R$ das partículas $A$ e $B$ sabendo que neste ambiente os as velocidades das partículas são afetadas por forças provenientes de campos externos que foram modeladas na equação $\vec{v}_R = 3\vec{v}_A - 4\vec{v}_B$. Qual o vetor unitário que determina a direção e o sentido de $\vec{v}_R$ nestas condições?

#### Exercício 7

Tudo é relativo! A amável leitora já deve ter ouvido esta frase. Uma mentira, das mais vis deste nossos tempos. Tudo é relativo, na física! Seria mais honesto. Não existe qualquer documento, artigo, livro, ou entrevista onde [Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) tenha dito tal sandice. Ainda assim, isso é repetido a exaustão. Não por nós. Nós dois estamos em busca da verdade do conhecimento. E aqui, neste ponto, entra o conceito de Einstein: as leis da física são as mesmas independente do observador. Isso quer dizer que, para entender um fenômeno, precisamos criar uma relação entre o observador e o fenômeno. Dito isso, considere que você está observando um trem que corta da direita para esquerda seu campo de visão em velocidade constante $\vec{V}_t = 10 \text{km/h}$. Nesse trem, um passageiro atravessa o vagão perpendicularmente ao movimento do trem em uma velocidade dada por $\vec{V}_p = 2 \text{km/h}$. Qual a velocidade deste passageiro para você, que está colocada de forma perfeitamente perpendicular ao movimento do trem?

#### Exercício 8

Vamos tornar o exercício 7 mais interessante: considere que você está observando um trem que corta da direita para esquerda seu campo de visão em velocidade constante $\vec{V}_t = 10 \text{km/h}$ subindo uma ladeira com inclinação de $25^\circ$. Nesse trem, um passageiro atravessa o vagão perpendicularmente ao movimento do trem em uma velocidade dada por $\vec{V}_p = 2 \text{km/h}$. Qual a velocidade deste passageiro para você, que está colocada de forma perfeitamente perpendicular ao movimento do trem?

### Vetores Posição e Distância

Um vetor posição, ou vetor ponto, é uma ferramenta útil para descrever a posição de um ponto no espaço em relação a um ponto de referência (geralmente a origem do sistema de coordenadas). Como uma flecha que começa na origem, o coração do sistema de coordenadas, onde $x$, $y$, e $z$ são todos zero, $(0,0,0)$, e termina em um ponto $P$ no espaço. Este ponto $P$ tem suas próprias coordenadas - digamos, $x$, $y$, e $z$.

O vetor posição $\vec{R}$ que vai da origem até este ponto $P$ será representado por $\vec{R}_P$. Se as coordenadas de $P$ são $(x, y, z)$, então o vetor posição $\vec{R}_P$ será:

$$\vec{R}_p = x\, \vec{a}_x + y\, \vec{a}_y + z\, \vec{a}_z$$

O que temos aprendido, na nossa jornada, até o momento, sobre vetores é simplesmente uma forma diferente de olhar para a mesma coisa. Sem nenhuma explicitação específica, estamos usando o conceito de Vetor Posição, desde que começamos este texto. 

A soma de vetores unitários, $\vec{a}_x$, $\vec{a}_y$, $\vec{a}_z$, que define um vetor em qualquer direção que escolhemos, sob um olhar alternativo irá definir o Vetor Posição de um dado ponto no espaço. Isso é possível porque, neste caso, estamos consideramos o vetor como uma seta que parte do zero - a origem - e se estende até qualquer ponto no espaço.

Como a doce leitora pode ver, está tudo conectado, cada parte fazendo sentido à luz da outra. Assim, aprenderemos a entender o espaço ao nosso redor, uma vetor de cada vez.

No universo dos problemas reais, onde estaremos sujeitos a forças na forma de gravidade, eletromagnetismo, ventos e correntes. Não podemos nos limitar a origem como ponto de partida de todos os vetores. Se fizermos isso, corremos o risco de tornar complexo o que é simples.

Na frieza da realidade, entre dois pontos quaisquer no espaço, $P$ e $Q$ será possível traçar um vetor. Um vetor que chamaremos de vetor distância e representaremos por $\vec{R}$.

Dois pontos no espaço, $P$ e $Q$, são como dois pontos num mapa. Cada um tem seu próprio vetor posição - seu próprio caminho da origem, o centro do mapa, até onde eles estão. Chamamos esses caminhos de $\vec{R}_P$ e $\vec{R}_Q$. Linhas retas que partem da origem, o centro do mapa e chegam a $P$ e $Q$. Usando para definir estes pontos os vetores posição a partir da origem.

Agora, se você quiser encontrar a distância entre $P$ e $Q$, não o caminho do centro do mapa até $P$ ou $Q$, mas o caminho direto partindo de $P$ até $Q$. Este caminho será o vetor distância $\vec{R}_{PQ}$.

Resta uma questão como encontramos $\vec{R}_{PQ}$?

Usamos a subtração de vetores. O vetor distância $\vec{R}_{PQ}$ será a diferença entre $\vec{R}_Q$ e $\vec{R}_P$. É como pegar o caminho de $Q$ ao centro do mapa, a origem do Sistema de Coordenadas Cartesianas, e subtrair o caminho de $P$ a este mesmo ponto. O que sobra é o caminho de $P$ até $Q$.

$$\vec{R} = \vec{R}_Q - \vec{R}_P$$

$\vec{R}$, a distância entre $P$ e $Q$, será geometricamente representado por uma seta apontando de $P$ para $Q$. O comprimento dessa seta é a distância entre $P$ e $Q$. Ou, em outras palavras, se temos um vetor, com origem em um ponto $P$ e destino em um ponto $Q$ a distância entre estes dois pontos será a magnitude deste vetor. E agora a amável leitora sabe porque chamamos de **vetor posição** ao vetor resultante a subtração entre dois vetores com origem no mesmo ponto. 

É um conceito simples, porém poderoso. Uma forma de conectar dois pontos em um espaço, uma forma de enxergar todo espaço a partir dos seus pontos e vetores. Definindo qualquer vetor a partir dos vetores posição. Bastando para tanto, definir um ponto comum para todo o espaço. Coisa que os sistemas de coordenadas fazem por nós graciosamente.

<p class="exp">
<b>Exemplo: 7</b><br>
Considerando que $P$ esteja nas coordenadas $(3,2,-1)$ e $Q$ esteja nas coordenadas $(1,-2,3)$. Logo, o vetor distância $\vec{R}_{PQ}$ será dado por: <br><br>
<b>Solução:</b>

\[\vec{R}_{PQ} = \vec{R}_P - \vec{R}_Q\]

Logo:

\[\vec{R}_{PQ} = (P_x-Q_x)\, \vec{a}_x + (P_y-Q_y)\, \vec{a}_y+(P_z-Q_z)\, \vec{a}_z\]

\[\vec{R}_{PQ} = (3-1)\, \vec{a}_x+(3-(-2))\, \vec{a}_y+((-1)-3)\, \vec{a}_z\]

\[\vec{R}_{PQ} = 2\, \vec{a}_x+5\, \vec{a}_y-4\, \vec{a}_z\]

</p>

<p class="exp">
<b>Exemplo 8:</b><br>
Dados os pontos $P_1(4,4,3)$, $P_2(-2,0,5)$ e $P_3(7,-2,1)$. (a) Especifique o vetor $\, \vec{a}$ que se estende da origem até o ponto $P_1$. (b) Determine um vetor unitário que parte da origem e atinge o ponto médio do segmento de reta formado pelos pontos $P_1$ e $P_2$. (c) Calcule o perímetro do triângulo formado pelos pontos $P_1$, $P_2$ e $P_3$.

<br><br>
<b>Solução:</b><br>
<b>(a)</b> o vetor $\, \vec{a}$ será o vetor posição do ponto $P_1(4,3,2)$ dado por:

$$\, \vec{a} = 4\, \vec{a}_x+4\, \vec{a}_y+3\, \vec{a}_z$$

<b>(b)</b> para determinar um vetor unitário que parte da origem e atinge o ponto médio do segmento de reta formado pelos pontos $P_1$ e $P_2$ precisamos primeiro encontrar este ponto médio $P_M$. Então:

\[P_M=\frac{P_1+P_2}{2} =\frac{(4,4,3)+(-2,0,5)}{2}\]

\[P_M=\frac{(2,4,8)}{2} = (1, 2, 4)\]

\[P_M=\, \vec{a}_x+2\, \vec{a}_y+4\, \vec{a}_z\]

Para calcular o vetor unitário na direção do vetor $P_M$ teremos:

\[\vec{a}\_{P_M}=\frac{(1, 2, 4)}{|(1, 2, 4)|} = \frac{(1, 3, 4)}{\sqrt{1^2+2^2+4^2} }\]

\[\vec{a}\_{P_M}=0.22\, \vec{a}_x+0.45\, \vec{a}_y+0.87\, \vec{a}_z\]

<b>(c)</b> finalmente, para calcular o perímetro do triângulo formado por: $P_1(4,4,3)$, $P_2(-2,0,5)$ e $P_3(7,-2,1)$, precisaremos somar os módulos dos vetores distância ente $P_1(4,3,2)$ e $P_2(-2,0,5)$, $P_2(-2,0,5)$ e $P_3(7,-2,1)$ e $P_3(7,-2,1)$ e $P_1(4,3,2)$.

\[\vert P_1P_2 \vert = \vert (4,4,3)-(-2,0,5) \vert = \vert (6,4,-2) \vert\]

\[\vert P_1P_2 \vert = \sqrt{6^2+4^2+2^2}=7,48\]

\[\vert P_2P_3 \vert = \vert (-2,0,5)-(7,-2,1) \vert = \vert (-9,2,-4) \vert\]

\[\vert P_2P_3 \vert = \sqrt{9^2+2^2+4^2}=10,05\]

\[\vert P_3P_1 \vert = \vert (7,-2,1)-(4,4,3) \vert = \vert (3,-6,-2) \vert\]

\[\vert P_3P_1 \vert = \sqrt{3^2+6^2+6^2}=7\]

Sendo assim o perímetro será:

\[\vert P_1P_2 \vert + \vert P_2P_3 \vert + \vert P_3P_1 \vert =7,48+10,05+7=24.53 \]
</p>

Vetores são como os ventos que cruzam o mar, invisíveis mas poderosos, guiando navios e direcionando correntes. Na matemática, eles têm sua própria linguagem, um código entre o visível e o invisível, mapeando direções e magnitudes. Aqui, você encontrará exercícios que irão desafiar sua habilidade de navegar por esse oceano numérico. Não são apenas problemas, mas bússolas que apontam para o entendimento mais profundo. Então pegue lápis e papel como se fossem um leme e um mapa e prepare-se para traçar seu próprio curso.

#### Exercício 9

Considere um sistema de referência onde as distâncias são dimensionadas apenas por unidades abstratas, sem especificação de unidades de medida. Nesse sistema, dois vetores são dados. O vetor $\vec{A}$ inicia na origem e termina no ponto $P$ com coordenadas $(8, -1, -5)$. Temos também um vetor unitário $\vec{c}$ que parte da origem em direção ao ponto $Q$, e é representado por $\frac{1}{3}(1, -3, 2)$. Se a distância entre os pontos $P$ e $Q$ é igual a 15 unidades, determine as coordenadas do ponto $Q$.

#### Exercício 10

Considere os pontos $P$ e $Q$ localizados em $(1, 3, 2)$ e $(4, 0, -1)$, respectivamente. Calcule: (a) O vetor posição $\vec{P}$; (b) O vetor distância de $P$ para $Q$, $\vec{PQ}$; (c) A distância entre $P$ e $Q$; (d) Um vetor paralelo a $\vec{PQ}$ com magnitude de 10.

### Produto Escalar

Há um jeito de juntar dois vetores - setas no espaço - e obter algo diferente: um número, algo mais simples, sem direção, sem sentido, direto e frio. Este é o Produto Escalar. **O resultado do Produto Escalar entre dois vetores é um valor escalar**.

A operação Produto Escalar recebe dois vetores e resulta em um número que, no espaço vetorial $\textbf{V}$, definido anteriormente, será um número real. Esse resultado tem algo especial: sua invariância. Não importa a orientação, rotação ou o giro que você imponha ao espaço vetorial, o resultado do Produto Escalar, continuará imutável, inalterado.

A amável leitora há de me perdoar, mas é preciso lembrar que escalares são quantidades que não precisam saber para onde estão apontando. Elas apenas são. Um exemplo? A Temperatura. Não importa como você oriente, gire ou mova um sistema de coordenadas aplicado no espaço para entender um fenômeno termodinâmico, a temperatura deste sistema permanecerá a mesma. A temperatura é uma quantidade que não tem direção nem sentido.

Aqui está o pulo da onça: enquanto um vetor é uma entidade direcionada, seus componentes são meros escalares. Ao decompor um vetor em seus componentes unitários — cada qual seguindo a direção de um eixo coordenado — é preciso entender que esses elementos são fluidos e mutáveis dependem das características do sistema de coordenadas. Os componentes se ajustam, se transformam e se adaptam quando você roda ou reorienta o espaço. Em contraste, o Produto Escalar, apesar de sua simplicidade, permanece constante, imperturbável às mudanças espaciais. Ele é um pilar invariável, vital para compreender tanto a estrutura do espaço quanto as dinâmicas que nele ocorrem.

Usando a linguagem da matemática, direta e linda, podemos dizer que dados os vetores $\vec{A}$ e $\vec{B}$, **o Produto Escalar entre $\vec{A}$ e $\vec{B}$ resultará em uma quantidade escalar**.  Esta operação será representada, usando a linguagem da matemática, por $\vec{A}\cdot \vec{B}$.

Aqui abro mão da isenção e recorro a geometria. Mais que isso, faremos uso da trigonometria para reduzir o Produto Escalar ao máximo de simplicidade usando uma equação que inclua o ângulo entre os dois vetores. Sem nos perdermos nas intrincadas transformações trigonométricas diremos que o Produto Escalar entre $\vec{A}$ e $\vec{B}$ será:

$$\vec{A} \cdot \vec{B} = \vert \vec{A}\vert \vert \vec{B} \vert cos(\theta_{AB})$$

Onde $\theta_{AB}$ representa o ângulo entre os dois vetores. Esta é a equação analítica do Produto Escalar. A ferramenta mais simples que podemos usar. Não é uma equação qualquer, ela representa a projeção do vetor $\vec{A}$ sobre o vetor $\vec{B}$. Se não, a paciente leitora, não estiver vendo esta projeção deve voltar a geometria, não a acompanharei nesta viagem, tenho certeza do seu sucesso. Em bom português dizemos que **o Produto Escalar entre dois vetores $\vec{A}$ e $\vec{B}$ quaisquer é o produto entre o produto das magnitudes destes vetores e o cosseno do menor ângulo entre eles**.

Vetores são como flechas atiradas no vazio do espaço. E como flechas, podem seguir diferentes caminhos.

Alguns vetores correm paralelos, como flechas lançadas lado a lado, nunca se encontrando. Eles seguem a mesma direção, compartilham o mesmo curso, mas nunca se cruzam. Sua jornada é sempre paralela, sempre ao lado. O ângulo entre eles, $\theta$, é $\text{zero}$ neste caso o cosseno entre eles, $cos(\theta)$ será então $1$. E o Produto Escalar entre eles será o resultado do produto entre suas magnitudes.

Outros vetores são transversais, como flechas que cortam o espaço em ângulos retos, ângulos $\theta = 90^\circ$. Eles não seguem a mesma direção, nem o mesmo caminho. Eles se interceptam, mas em ângulos precisos, limpos, cortando o espaço como uma grade. O cosseno entre estes vetores é $0$. E o Produto Escalar será zero independente das suas magnitudes.

Entre os vetores que correm em paralelo e aqueles que se cruzam transversalmente estão os limites superior e inferior do Produto Escalar, seu valor máximo e mínimo. Estes são os vetores que se cruzam em qualquer ângulo, como flechas lançadas de pontos distintos, cruzando o espaço de formas únicas. Eles podem se encontrar, cruzar caminhos em um único ponto, ou talvez nunca se cruzem. Estes vetores desenham no espaço uma dança de possibilidades, um balé de encontros e desencontros. Aqui, o cosseno não pode ser determinado antes de conhecermos os vetores em profundidade. Para estes rebeldes, usamos o **ângulo mínimo** entre eles. Um ângulo agudo. Quando dois vetores se cruzam, dois ângulos são criados. Para o Produto Escalar usaremos sempre o menor deles, o ângulo, mínimo, interno deste relacionamento.

Como flechas no espaço, vetores desenham caminhos - paralelos, transversais ou se cruzando em qualquer ângulo. Vetores são a linguagem das forças no espaço, a escrita das distâncias e direções. Eles são os contadores de histórias do espaço tridimensional.

A matemática da Álgebra Vetorial destila estes conceitos simplesmente como: se temos um vetor $\, \vec{a}$ e um vetor $\vec{B}$ teremos o Produto Escalar entre eles dado por:

$$\vec{A}\cdot \vec{B} = A_xB_x+ A_yB_y+ A_zB_z$$

Seremos então capazes de abandonar a equação analítica, e voltarmos aos mares tranquilos de ventos suaves da Álgebra Linear. A matemática nos transmite paz e segurança. Exceto quando estamos aprendendo. Nestes momentos, nada como uma xícara de chá morno e um exemplo para acender a luz do entendimento.

<p class="exp">
<b>Exemplo 9:</b><br>
Dados os vetores $\vec{A}=3\, \vec{a}_x + 4\, \vec{a}_y + \, \vec{a}_z$ e $\vec{B}=\, \vec{a}_x+2\, \vec{a}_y-5\, \vec{a}_z$ encontre o ângulo $\theta$ entre $\, \vec{a}$ e $\vec{B}$.
<br><br>
<b>Solução:</b><br>
Para calcular o ângulo vamos usar a equação analítica do Produto Escalar:

\[\vec{A}\cdot \vec{B} =\vert \vec{a}\vert \vert \vec{B} \vert cos(\theta)\]

Precisaremos dos módulos dos vetores e do Produto Escalar entre eles. Calculando o Produto Escalar a partir dos componentes vetoriais de cada vetor teremos:

\[\vec{A}\cdot \vec{B} = (3,4,1)\cdot(1,2,-5) \]

\[\vec{A}\cdot \vec{B} = (3)(1)+(4)(2)+(1)(-5)=6\]

Calculando os módulos de $\vec{A}$ e $\vec{B}$, teremos:

\[ \vert \vec{A} \vert = \vert (3,4,1) \vert =\sqrt{3^2+4^2+1^2}=5,1\]

\[ \vert \vec{B} \vert = \vert (1,2,-5) \vert =\sqrt{1^2+2^2+5^2}=5,48\]

Já que temos o Produto Escalar e os módulos dos vetores podemos aplicar nossa equação analítica:

\[ \vec{A}\cdot \vec{B} =\vert \vec{A}\vert \vert \vec{B} \vert cos(\theta)\]

logo:

\[ 6 =(5,1)(5,48)cos(\theta) \therefore cos(\theta) = \frac{6}{27,95}=0,2147 \]

\[ \theta = arccos(0,2147)=77,6^\circ \]

</p>

Até agora, estivemos estudando um espaço de três dimensões, traçando vetores que se projetam em comprimentos, larguras e alturas do Espaço Cartesiano. Isso serve para algumas coisas. Para resolver alguns dos problemas que encontramos na dança de forças e campos que tecem o tecido do mundo físico. Mas nem sempre é o bastante.

A verdade é que o universo é mais complexo do que as três dimensões que podemos tocar e ver. Há mundos além deste, mundos que não podemos ver, não podemos tocar, mas podemos imaginar. Para esses mundos, precisamos de mais. Muito mais.

Álgebra vetorial é a ferramenta que usamos para desenhar mundos. Com ela, podemos expandir nosso pensamento para além das três dimensões, para espaços de muitas dimensões. Espaços que são mais estranhos, mais complicados, mas também mais ricos em possibilidades. Talvez seja hora de reescrever nossa definição de Produto Vetorial, a hora de expandir horizontes. Não apenas para o espaço tridimensional, mas para todos os espaços que podem existir. Isso é o que a álgebra vetorial é: uma linguagem para desenhar mundos, de três dimensões ou mais.

Generalizando o Produto Escalar entre dois vetores $\vec{A}$ e $\vec{B}$ com $N$ dimensões teremos:

$$\vec{A} \cdot \vec{B} = \sum\limits_{i=1}\limits^{N} \vec{A}_i\vec{b}_i$$

Onde $i$ é o número de dimensões. Assim, se $i=3$ e chamarmos estas dimensões $x$, $y$, $z$ respectivamente para $i=1$, $i=2$ e $i=3$ teremos:

$$\vec{A} \cdot \vec{B} = \sum\limits_{i=1}\limits^{3} \vec{a}_i\vec{b}_i = a_1b_1 +a_2b_2 + a_3b_3 $$

Ou, substituindo os nomes das dimensões:

$$\vec{A} \cdot \vec{B} = \, \vec{a}_x\vec{b}_x +\, \vec{a}_y\vec{b}_y + \, \vec{a}_z\vec{b}_z $$

Não vamos usar dimensões maiores que $3$ neste estudo. Contudo, achei que a gentil leitora deveria perceber esta generalização. No futuro, em outras disciplinas, certamente irá me entender.

#### Exercício 11

Em um novo projeto de engenharia civil para a construção de uma estrutura triangular inovadora, foram demarcados três pontos principais para as fundações. Esses pontos, determinados por estudos topográficos e geotécnicos, foram identificados como $A(4, 0, 3)$, $B(-2, 3, -4)$ e $C(1, 3, 1)$ em um espaço tridimensional utilizando o Sistema de Coordenadas Cartesianas. A equipe de engenheiros precisa compreender a relação espacial entre esses pontos, pois isto impacta diretamente na distribuição das cargas e na estabilidade da estrutura.

Seu desafio será determinar o o ângulo $\theta_{BAC}$ entre estes vetores crucial para a análise estrutural, pois determina o direcionamento das forças na fundação.

Uma vez que tenhamos entendido a operação Produto Escalar, nos resta entender suas propriedades:

1. **Comutatividade:** o Produto Escalar tem uma beleza simples quase rítmica. Como a batida de um tambor ou o toque de um sino, ele se mantém o mesmo não importa a ordem. Troque os vetores - a seta de $\vec{A}$ para $\vec{B}$ ou a flecha de $\vec{B}$ para $\vec{A}$ - e você obtém o mesmo número, o mesmo escalar. Isso é o que significa ser comutativo. Ou seja: $\vec{A} \cdot \vec{B} = \vec{B}\cdot \vec{A}$

2. **Distributividade em Relação a Adição:** o Produto Escalar também é como um rio dividindo-se em afluentes. Você pode distribuí-lo, espalhá-lo, dividir um vetor por muitos. Adicione dois vetores e multiplique-os por um terceiro - você pode fazer isso de uma vez ou pode fazer um por vez. O Produto Escalar não se importa. Ele dá o mesmo número, a mesma resposta. Isso é ser distributivo em relação a adição. Dessa forma teremos: $\vec{A}\cdot (\vec{B}+\vec{C}) = \vec{A}\cdot \vec{B} + \vec{A}\cdot \vec{C}$.

3. **Associatividade com Escalares:** o Produto Escalar é como um maestro habilidoso que sabe equilibrar todos os instrumentos em uma orquestra. Imagine um escalar como a intensidade da música: aumente ou diminua, e a harmonia ainda será mantida. Multiplicar um vetor por um escalar e, em seguida, realizar o Produto Escalar com outro vetor é o mesmo que primeiro executar o Produto Escalar e depois ajustar a intensidade. O Produto Escalar, em sua elegância matemática, garante que o show continue de maneira harmoniosa, independentemente de quando a intensidade é ajustada. Essa é a essência da associatividade com escalares. Portanto, podemos dizer que: $k(\vec{A} \cdot \vec{B}) = (k \vec{A}) \cdot \vec{B} = \vec{A} \cdot (k\vec{B})$

4. **Produto Escalar do Vetor Consigo Mesmo:** O Produto Escalar tem um momento introspectivo, como um dançarino girando em um reflexo de espelho. Quando um vetor é multiplicado por si mesmo, ele revela sua verdadeira força, sua magnitude ao quadrado. É uma dança solitária, onde o vetor se alinha perfeitamente consigo mesmo, na mais pura sintonia. Esta auto-referência nos mostra o quanto o vetor se projeta em sua própria direção, revelando a essência de sua magnitude. Assim, temos: $\vec{A} \cdot \vec{A} = \vert \vec{A} \vert^2$. Veja um vetor $\vec{A}$. Uma seta solitária estendendo-se no espaço. Imagine colocar outra seta exatamente igual, exatamente no mesmo lugar. Duas Setas juntas, $\vec{A}$ e $\vec{A}$, sem nenhum ângulo entre elas.

Por que? Porque o ângulo $\theta$ entre um vetor e ele mesmo é $zero$. E o cosseno de zero é $1$. Assim:

$$\vec{A}\cdot \vec{A} = \vert \vec{A} \vert^2$$

Para simplificar, vamos dizer que $\vec{A}^2$ é o mesmo que $ \vert \vec{A} \vert ^2$. Uma notação, uma abreviação para o comprimento, magnitude, de $\vec{A}$ ao quadrado. Aqui está a lição: **um vetor e ele mesmo, lado a lado, são definidos pela magnitude do próprio vetor, ao quadrado**. É um pequeno pedaço de sabedoria, um truque, uma ferramenta. Mantenha esta ferramenta sempre à mão, você vai precisar.

Assim como as ondas em uma praia, indo e voltando, de tempos em tempos precisamos rever as ferramentas que adquirimos e o conhecimento que construímos com elas. Em todos os sistemas de coordenadas que usamos para definir o espaço $\mathbf{V}$ os vetores unitários são ortogonais. Setas no espaço que se cruzam em um ângulo reto. Este ângulo reto garante duas propriedades interessantes.

$$\vec{a}_x\cdot \, \vec{a}_y=\, \vec{a}_x\cdot \, \vec{a}_z=\, \vec{a}_y\cdot \, \vec{a}_z=0$$

$$\vec{a}_x\cdot \, \vec{a}_x=\, \vec{a}_y\cdot \, \vec{a}_y=\, \vec{a}_z\cdot \, \vec{a}_z=1$$

A primeira garante que o Produto Escalar entre quaisquer dois componentes vetoriais ortogonais é $zero$, a segunda que o Produto Escalar entre os mesmos dois componentes vetoriais é $1$. Essas são duas verdades que podemos segurar firmes enquanto navegamos pelo oceano do espaço vetorial. Como um farol em uma noite tempestuosa, elas nos guiarão e nos ajudarão a entender o indescritível. Mais que isso, serão as ferramentas que usaremos para transformar o muito difícil em muito fácil.

Desculpe-me! Esta ambição que me força a olhar além me guia aos limites do possível. Assim como expandimos o número de dimensões para perceber que o impacto do Produto Vetorial se estende além dos limites da nossa, precisamos, novamente, levar as dimensões do nosso universo ao ilimitável.

As propriedades derivadas da ortogonalidade dos componentes dos sistemas de coordenadas podem ser expressas usando o [Delta de Kronecker](https://en.wikipedia.org/wiki/Kronecker_delta) definido por [Leopold Kronecker](https://en.wikipedia.org/wiki/Leopold_Kronecker)(1823–1891). O Delta de Kronecker é uma forma de representar por índices as dimensões do espaço vetorial, uma generalização, para levarmos a Álgebra Linear ao seu potencial máximo, sem abandonar os limites que definimos para o estudo do Eletromagnetismo. sem delongas, teremos:

$$
\begin{equation}
  \delta_{\mu \upsilon}=\begin{cases}
    1, se \space\space \mu = \upsilon .\\
    0, se \space\space \mu \neq \upsilon.
  \end{cases}
\end{equation}
$$

Usando o Delta de Kronecker podemos escrever as propriedades dos componentes ortogonais unitários em relação ao Produto Escalar como:

$$\vec{a}_\mu \cdot \, \vec{a}_\upsilon = \delta_{\mu \upsilon}$$

Que será útil na representação computacional de vetores e no entendimento de transformações vetoriais em espaços com mais de $3$ dimensões. Que, infelizmente, estão além deste ponto na nossa jornada. Não se deixe abater, ficaremos limitados a $3$ dimensões. Contudo, não nos limitaremos ao Produto Escalar. Outras maravilhas virão.

<p class="exp">
<b>Exemplo 10:</b><br>
Dados os vetores $\vec{A} = (3, 2, 1)$ e $\vec{B} = (1, -4, 2)$, calcule o Produto Escalar $\vec{A} \cdot \vec{B}$ e também $\vec{B} \cdot \vec{A}$. Verifique a propriedade da comutatividade.
<br><br>
<b>Solução:</b><br>
Tudo que precisamos para provar a comutatividade é fazer o Produto Escalar em duas ordens diferentes em busca de resultados iguais.
   \[ \vec{A} \cdot \vec{B} = 3 \times 1 + 2 \times (-4) + 1 \times 2 = 3 - 8 + 2 = -3 \]  

   \[ \vec{B} \cdot \vec{A} = 1 \times 3 + (-4) \times 2 + 2 \times 1 = 3 - 8 + 2 = -3\]  
</p>

<p class="exp">
<b>Exemplo 11:</b><br>
Dados os vetores $\vec{A} = (2, 3, 1)$, $\vec{B} = (1, 2, 0)$ e $\vec{C} = (3, 1, 3)$, calcule $\vec{A} \cdot (\vec{B} + \vec{C})$ e compare com $\vec{A} \cdot \vec{B} + \vec{A} \cdot \vec{C}$.
<br><br>
<b>Solução:</b><br>

Primeiro, encontre $\vec{B} + \vec{C} = (1+3, 2+1, 0+3) = (4, 3, 3)$.  

   \[ \vec{A} \cdot (\vec{B} + \vec{C}) = 2 \times 4 + 3 \times 3 + 1 \times 3 = 8 + 9 + 3 = 20\]  
   \[ \vec{A} \cdot \vec{B} + \vec{A} \cdot \vec{C} = 2 \times 1 + 3 \times 2 + 1 \times 0 + 2 \times 3 + 3 \times 1 + 1 \times 3\]  
   \[ \vec{A} \cdot \vec{B} + \vec{A} \cdot \vec{C} = 2 + 6 + 0 + 6 + 3 + 3\]
   \[ \vec{A} \cdot \vec{B} + \vec{A} \cdot \vec{C} = 20\]
</p>
  
#### Exercício 12

Considere o vetor $\vec{F} = (x, y, z)$ perpendicular ao vetor $\vec{G} = (2, 3, 1)$. Sabendo que $\vec{F} \cdot \vec{F} = 9$. Determine os componentes que definem o vetor $\vec{F}$.

#### Exercício 13

Calcule o Produto Escalar de $\vec{C} = \vec{A} - \vec{B}$ com ele mesmo.

### Produto Vetorial

Imagine dois vetores, $\vec{A}$ e $\vec{B}$, como setas lançadas no espaço. Agora, imagine desenhar um paralelogramo com as magnitudes de $\vec{A}$ e $\vec{B}$ como lados. O Produto Vetorial de $A$ e $B$, representado por $\vec{A} \times \vec{B}$, é como uma seta disparada diretamente para fora desse paralelogramo, tão perfeitamente perpendicular quanto um mastro em um navio.

**A magnitude, o comprimento dessa seta, é a área do paralelogramo formado por $\vec{A}$ e $\vec{B}$**. É um número simples, mas importante. Descreve o quão longe a seta resultante da interação entre $\vec{A}$ e $\vec{B}$ se estende no espaço. O comprimento do vetor resultado do Produto Vetorial. **O resultado do Produto Vetorial entre dois vetores é um vetor.**

imagine que temos dois vetores, firme e diretos, apontando em suas direções particulares no espaço. Chamamos eles de $\vec{A}$ e $\vec{B}$. Esses dois, em uma dança matemática, se entrelaçam em um Produto Vetorial, formando um terceiro vetor, o $\vec{C}$, perpendicular a ambos $\vec{A}$ e $\vec{B}$. Mais que isso, perpendicular ao paralelogramo formado por $\vec{A}$ e $\vec{B}$. Ainda mais, perpendicular ao plano formado por $\vec{A}$ e $\vec{B}$. Esta é a característica mais marcante do Produto Vetorial.

Portanto, a dança do Produto Vetorial é peculiar e intrigante, os dançarinos não trocam de lugar como a dança tradicional e a sequência de seus passos importa, mesmo assim ela acolhe a velha regra da distributividade. Uma dança peculiar no palco da matemática. Que leva a criação de uma novo dançarino, um novo vetor, perpendicular ao plano onde dançam os vetores originais. Esse novo vetor, esse Produto Vetorial, pode ser definido por uma equação analítica, geométrica, trigonométrica:

$$A \times B = \vert A \vert \vert B \vert sen(\theta_{AB}) a_n$$

Onde $a_n$ representa o vetor unitário na direção perpendicular ao plano formado pelo paralelogramo formado por $A$ e $B$.
É uma fórmula simples, mas poderosa. Ela nos diz como calcular o Produto Vetorial, como determinar a direção, o sentido e a intensidade desta seta, lançada ao espaço.

A direção dessa seta, representada pelo vetor unitário $a_n$, será decidida pela regra da mão direita. Estenda a mão, seus dedos apontando na direção de $A$. Agora, dobre seus dedos na direção de $B$. Seu polegar, erguido, aponta na direção de $a_n$, na direção do Produto Vetorial.

O Produto Vetorial determina uma forma de conectar dois vetores, $A$ e $B$, e criar algo novo: um terceiro vetor, lançado diretamente para fora do plano criado por $A$ e $B$. E esse vetor, esse Produto Vetorial, tem tanto uma magnitude - a área do paralelogramo - quanto uma direção - decidida pela regra da mão direita. É uma forma de entender o espaço tridimensional. E como todas as coisas na álgebra vetorial, é simples, mas poderoso.

$$\vec{A} \times \vec{A} = \vert \vec{A} \vert  \vert \vec{B} \vert sen(\theta_{AB}) a_n$$

É uma equação poderosa e simples, útil, muito útil, mas geométrica, trigonométrica e analítica. Algebricamente o Produto Vetorial pode ser encontrado usando uma matriz. As matrizes são os sargentos do exército da Álgebra Vetorial, úteis mas trabalhosas e cheias de regras. Considerando os vetores $\vec{a}=\, \vec{a}_x \, \vec{a}_x+\, \vec{a}_y \, \vec{a}_y+\, \vec{a}_z \, \vec{a}_z$ e $\vec{B}=B_x \, \vec{a}_x+B_y \, \vec{a}_y+B_z \, \vec{a}_z$ o Produto Vetorial $\vec{A}\times \vec{B}$ será encontrado resolvendo a matriz:

$$
\vec{A}\times \vec{B}=\begin{vmatrix}
\vec{a}_x & \, \vec{a}_y & \, \vec{a}_z\\
\vec{a}_x & \, \vec{a}_y & \, \vec{a}_z\\
B_x & B_y & B_z
\end{vmatrix}
$$

A matriz será sempre montada desta forma. A primeira linha om os vetores unitários, a segunda com o primeiro operando, neste caso os componentes de $\vec{A}$ e na terceira com os componentes de $\vec{B}$. A Solução deste produto será encontrada, mais facilmente com o Método dos Cofatores. Para isso vamos ignorar a primeira linha.

Ignorando também a primeira coluna, a coluna do vetor unitário $\vec{a}_x$ resta uma matriz composta de:

$$
\begin{vmatrix}
\vec{a}_y & \, \vec{a}_z\\
B_y & B_z
\end{vmatrix}
$$

O Esta matriz multiplicará o vetor unitário $\vec{a}_x$. Depois vamos construir outras duas matrizes como esta. A segunda será encontrada quando ignorarmos a coluna referente ao unitário $\vec{a}_y$, que multiplicará o oposto do vetor $\vec{a}_y$.

$$
\begin{vmatrix}
\vec{a}_x & \, \vec{a}_z\\
B_x & B_z
\end{vmatrix}
$$

Finalmente ignoramos a coluna referente ao vetor unitário $\vec{a}_z$ para obter:

$$
\begin{vmatrix}
\vec{a}_x & \, \vec{a}_y\\
B_x & B_y
\end{vmatrix}
$$

Que será multiplicada por $\, \vec{a}_z$. Colocando tudo junto, em uma equação matricial teremos:

$$
\vec{A}\times \vec{B}=\begin{vmatrix}
\vec{a}_x & \, \vec{a}_y & \, \vec{a}_z\\
\vec{a}_x & \, \vec{a}_y & \, \vec{a}_z\\
B_x & B_y & B_z
\end{vmatrix}=\begin{vmatrix}
\vec{a}_y & \, \vec{a}_z\\
B_y & B_z
\end{vmatrix}\, \vec{a}_x-\begin{vmatrix}
\vec{a}_x & \, \vec{a}_z\\
B_x & B_z
\end{vmatrix}\, \vec{a}_y+\begin{vmatrix}
\vec{a}_x & \, \vec{a}_y\\
B_x & B_y
\end{vmatrix}\, \vec{a}_z
$$

Cuide o negativo no segundo termo como cuidaria do leme do seu barco, sua jornada depende disso e o resultado do Produto Vetorial Também. Uma vez que a equação matricial está montada. Cada matriz pode ser resolvida usando a [Regra de Sarrus](https://en.wikipedia.org/wiki/Rule_of_Sarrus) que, para matrizes de $2\times 2$ se resume a uma multiplicação cruzada. Assim, nosso Produto Vetorial será simplificado por:

$$\vec{A}\times \vec{B}=(\vec{a}_y B_z- \, \vec{a}_z B_y)\, \vec{a}_x-(\vec{a}_x B_z-\, \vec{a}_z B_x)\,\vec{a}_y+(\vec{a}_x B_y-\, \vec{a}_y B_x)\, \vec{a}_z$$

Cuidado com os determinantes, o Chapeleiro não era louco por causa do chumbo, muito usado na fabricação de chapéus quando [Lewis Carroll](https://en.wikipedia.org/wiki/Lewis_Carroll) escreveu as histórias de Alice. Ficou louco [resolvendo determinantes](https://www.johndcook.com/blog/2023/07/10/lewis-carroll-determinants/). Talvez um exemplo afaste a insanidade tempo suficiente para você continuar estudando eletromagnetismo.

<p class="exp">
<b>Exemplo 12:</b><br>
Dados os vetores $\vec{A}=\, \vec{a}_x+2\, \vec{a}_y+3\, \vec{a}_z$ e $\vec{B}=4\, \vec{a}_x+5\, \vec{a}_y-6\, \vec{a}_z$. (a) Calcule o Produto Vetorial entre $\vec{A}$ e $\vec{B}$. (b) Encontre o ângulo $\theta$ entre $\vec{A}$ e $\vec{B}$.
<br><br>
<b>Solução:</b><br>
(a) Vamos começar com o Produto Vetorial:

\[
\vec{A}\times \vec{B}=\begin{vmatrix}
\vec{a}_x & \, \vec{a}_y & \, \vec{a}_z\\
\vec{a}_x & \, \vec{a}_y & \, \vec{a}_z\\
B_x & B_y & B_z \end{vmatrix} = \begin{vmatrix}
\vec{a}_x & \, \vec{a}_y & \, \vec{a}_z\\
1 & 2 & 3\\
4 & 5 & -6
\end{vmatrix}
\]

Que será reduzida a:

\[
\vec{A}\times \vec{B} = \begin{vmatrix}
2 & 3\\
5 & -6
\end{vmatrix}\, \vec{a}_x - \begin{vmatrix}
1 & 3\\
4 & -6
\end{vmatrix}\, \vec{a}_y + \begin{vmatrix}
1 & 2\\
4 & 5
\end{vmatrix}\, \vec{a}_z
\]

Usando Sarrus em cada uma destas matrizes teremos:

\[\vec{A} \times \vec{B} = (2(-6) - 3(5)) \, \vec{a}_x - (1(-6)-3(4)) \, \vec{a}_y + (1(5)-2(4)) \, \vec{a}_z\]

\[\vec{A} \times \vec{B} = -27 \, \vec{a}_x + 18 \, \vec{a}_y - 3 \, \vec{a}_z\]

Esta foi a parte difícil, agora precisamos dos módulos, magnitudes, dos vetores $\vec{A}$ e $\vec{B}$.

\[\vert \vec{A} \vert = \sqrt{1^2+2^2+3^2} = \sqrt{14} \approx 3.74165\]

\[\vert \vec{B} \vert = \sqrt{4^2+5^2+6^2} = \sqrt{77} \approx 8.77496\]

Para calcular o ângulo vamos usar a equação analítica, ou trigonométrica, do Produto Vetorial:

\[\vec{A} \times \vec{B} = \vert \vec{A} \vert  \vert \vec{B} \vert sen(\theta_{AB}) a_n\]

A forma mais fácil de resolver este problema é aplicar o módulo aos dois lados da equação. Se fizermos isso, teremos:

\[\vert \vec{A} \times \vec{B} \vert = \vert \vec{A} \vert \vert \vec{B} \vert sen(\theta_{AB}) \vert a_n \vert \]

Como $a_n$ é um vetor unitário, por definição $\vert a_n \vert = 1$ logo:

\[\vert \vec{A} \times \vec{B} \vert = \vert \vec{A} \vert \vert \vec{B} \vert sen(\theta_{AB})\]

Ou, para ficar mais claro:

\[sen(\theta_{AB}) = \frac{\vert \, \vec{a} \times \vec{B} \vert}{\vert \, \vec{a} \vert \vert \vec{B} \vert}\]

Os módulos de $\vec{A}$ e $\vec{B}$ já tenos, precisamos apenas do módulo de $\vec{A}\times \vec{B}$.

\[
\vert \vec{A}\times \vec{B} \vert = \sqrt{27^2+16^2+3^2} = \sqrt{994} \approx 31.5298
\]

Assim o seno do ângulo $\theta_{AB}$ será dado por:

\[sen(\theta_{AB}) = \frac{\sqrt{994}}{(\sqrt{14})(\sqrt{77})} \approx \frac{31.5298}{(3.74165)(8.77496)}\]

\[sen(\theta_{AB}) = 0.960316\]

\[ \theta_{AB} =73.8^\circ \]

</p>

O Produto Vetorial é como uma dança entre vetores. E como todas as danças tem características únicas e interessantes expressas na forma de propriedades matemáticas:

1. **Comutatividade:** no universo dos vetores, há uma dança estranha acontecendo. $\vec{A} \times \vec{B}$ e $\vec{B} \times \vec{A}$ não são a mesma coisa, eles não trocam de lugar facilmente como dançarinos em um salão de baile. Em vez disso, eles são como dois boxeadores em um ringue, um o espelho do outro, mas em direções opostas. Assim, $\vec{A} \times \vec{B}$ é o oposto de $\vec{B} \times \vec{A}$. Assim, **O Produto Vetorial não é comutativo**: 

   $$\vec{A} \times \vec{B} =-\vec{B} \times \vec{A}$$

2. **Associatividade:** imagine três dançarinos: $\vec{A}$, $\vec{B}$ e $\vec{C}$. A sequência de seus passos importa. $\vec{A}$ dançando com $\vec{B}$, depois com $\vec{C}$, não é o mesmo que $\vec{A}$ dançando com o resultado de $\vec{B}$ e $\vec{C}$ juntos. Assim como na dança, a ordem dos parceiros importa. **O Produto Vetorial não é associativo**. Desta forma:

   $$\vec{A} \times (\vec{B} \times \vec{C}) \neq (\vec{A} \times \vec{B}) \times \vec{C}$$

3. **Distributividade:** existe um aspecto familiar. Quando $\vec{A}$ dança com a soma de $\vec{B}$ e $\vec{C}$, é a mesma coisa que $\vec{A}$ dançando com $\vec{B}$ e depois com $\vec{C}$. **O Produto Vetorial é distributivo**. A distributividade, uma velha amiga, nos conhecemos deste dos tempos da aritmética, aparece aqui, guiando a dança. O que pode ser escrito como:

   $$\vec{A} \times (\vec{B}+\vec{C}) = \vec{A} \times \vec{B} + \vec{A} \times \vec{C}$$

4. **Multiplicação por Escalar:** agora entra em cena um escalar, $k$, um número simples, porém carregado de influência. Ele se aproxima do Produto Vetorial e o muda, mas não de maneira selvagem ou imprevisível, e sim com a precisão de um relojoeiro. A magnitude do Produto Vetorial é esticada ou contraída pelo escalar, dependendo de seu valor. Isto pode ser escrito matematicamente como:

   $$k(A \times B) = (kA) \times B = A \times (kB)$$

   Porém, como o norte em uma bússola, a direção do Produto Vetorial não se altera. O resultado é um novo vetor, $\vec{D}$, que é um múltiplo escalar do original $\vec{C}$. O vetor $\vec{D}$ carrega a influência do escalar $k$, mas mantém a orientação e sentido originais de $\vec{C}$ para todo $k >0$.

5. **Componentes Unitários**: por fim, precisamos tirar para dançar os vetores unitários. Estrutura de formação dos nossos sistemas de coordenadas. Como Produto Vetorial $\vec{A}\times \vec{B}$ produz um vetor ortogonal ao plano formado por $\vec{A}$ e $\vec{B}$ a aplicação desta operação a dois dos vetores unitários de um sistema de coordenadas irá produzir o terceiro vetor deste sistema. Observando o Sistema de Coordenadas Cartesianas teremos:

  $$\vec{a}_x\times \, \vec{a}_y = \, \vec{a}_z$$

  $$\vec{a}_x\times \, \vec{a}_z = \, \vec{a}_y$$

  $$\vec{a}_y\times \, \vec{a}_z = \, \vec{a}_x$$

  Esta propriedade do Produto Vetorial aplicado aos componentes de um vetor é mais uma ferramenta que precisamos manter à mão. Um conjunto de regras que irão simplificar equações e iluminar o desconhecido de forma quase natural.

#### Exercício 14

Considerando a equação analítica do Produto escalar, $\vec{A} \cdot \vec{B} = \vert \vec{A} \vert \vert \vec{B} \vert cos(\theta)$, e a equação analítica do Produto Vetorial, $\vec{A} \times \vec{A} = \vert \vec{A} \vert \vert \vec{B} \vert sen(\theta_{AB})$ prove que estas duas operações são distributivas.

### Produto Triplo Escalar

O Produto Triplo Escalar é um conceito matemático definido para operação de três vetores $\vec{A}$, $\vec{B}$, e $\vec{C}$ em um espaço tridimensional e será determinado por:

$$\vec{A} \cdot (\vec{B} \times \vec{C})$$

A equação tem suas raízes nas obras de matemáticos como [Augustin-Louis Cauchy](https://en.wikipedia.org/wiki/Augustin-Louis_Cauchy) e [Josiah Willard Gibbs](https://en.wikipedia.org/wiki/Josiah_Willard_Gibbs), que contribuíram para o desenvolvimento da álgebra vetorial. A notação de Gibbs é particularmente útil para representar produtos escalares e vetoriais, bem como outras operações vetoriais. Segundo a notação de Gibbs, vetores são presentados por letras latinas em negrito $\mathbf{A}$ ou por letras com setas $\vec{A}$. Além disso, as notações que usamos para representar os produtos escalares e vetoriais, também foram desenvolvidas por Gibbs. Também devemos a Gibbs as representações das operações gradiente $\nabla$, o divergente, $\nabla \cdot \vec{F}$ e o rotacional, $\nabla \times \vec{F}$ do cálculo infinitesimal de campos vetoriais. Já Cauchy estudou a Álgebra Vetorial. Sua pesquisa com determinantes ajudou a definir o uso de determinantes para a solução de operações vetoriais.

O Produto Triplo Escalar pode ser expresso em termos de determinantes de matrizes. Se você tem três vetores $\vec{A} = [a_x, a_y, a_z]$, $\vec{B} = [b_x, b_y, b_z]$, e $\vec{C} = [c_x, c_y, c_z] \), o Produto Triplo Escalar $\vec{A} \cdot (\vec{B} \times \vec{C})$ será igual ao determinante da matriz $3x3$ formada pelos componentes desses vetores e dada por:

$$
\vec{A} \cdot (\vec{B} \times \vec{C}) = \begin{vmatrix}
a_x & a_y & a_z \\
b_x & b_y & b_z \\
c_x & c_y & c_z
\end{vmatrix}
$$

Cuja solução pode ser encontrada com as mesmas técnicas que usamos para resolver o Produto Vetorial. O Produto Triplo Escalar é uma extensão natural do produto vetorial e do produto escalar, e é uma ferramenta útil em várias áreas da ciência e da engenharia, incluindo física, eletromagnetismo, engenharia mecânica, e ciência da computação. A Leitora é de concordar que é uma operação simples. E como tudo que é simples, merece atenção.

O vetor $\vec{B} \times \vec{C}$ representa a área de um paralelogramo cujos lados serão dados por $\vec{B}$ e $\vec{C}$. Logo, $\vec{A} \cdot (\vec{B} \times \vec{C})$ representa essa área multiplicada pelo componente de $\vec{A}$ perpendicular a esta área o que resulta em um volume. Neste caso, o volume é o mesmo, não importa como você combine os vetores $\vec{A}$, $\vec{B}$, e $\vec{C $, no Produto Triplo Escalar com exceção de:

$$
\vec{A} \cdot (\vec{B} \times \vec{C}) = -\vec{A} \cdot (\vec{C} \times \vec{B})
$$

## PRECISA REESCREVER PARA INCLUIR O CONCEITO DA REGRA DA MÃO DIREITA

Portanto, o valor do Produto Triplo Escalar $\vec{A} \cdot (\vec{B} \times \vecCA})$ é positivo se os vetores $\vec{A}$, $\vec{B}$, e $\vec{C}$ são organizados de tal forma que $\vec{A}$ está no mesmo lado do plano formado por $\vec{B}$ e $\vec{B}$ como indicado pela regra da mão direita ao rotacionar $\vec{B}$ para $\vec{C}$. O valor será negativo se $\vec{A}$ está no lado oposto deste plano. O produto triplo permanece inalterado se os operadores de produto escalar e vetorial forem trocados:

$$
\vec{A} \cdot (\vec{B} \times \vec{C}) = \vec{A} \times (\vec{B} \cdot \vec{C})
$$

O Produto Triplo Escalar também é invariante sob qualquer permutação cíclica de $\vec{A}$, $\vec{B}$, e $\vec{C}$,

$$
\vec{A} \cdot (\vec{B} \times \vec{C}) = \vec{B} \cdot (\vec{C} \times \vec{A}) = \vec{C} \cdot (\vec{A} \times \vec{B})
$$

Contudo qualquer permutação anti-cíclica faz com que o Produto Triplo Escalar mude de sinal,

$$
\vec{A} \cdot (\vec{B} \times \vec{C}) = -\vec{B} \cdot (\vec{A} \times \vec{C})
$$

O Produto Triplo Escalar é zero se quaisquer dois dos vetores $\vec{A}$, $\vec{B}$, e $\vec{C}$ são paralelos, ou se $\vec{A}$, $\vec{B}$, e $\vec{C}$ são coplanares. Se $\vec{A}$, $\vec{B}$, e $\vec{C}$ não são coplanares, então qualquer vetor $\vec{R}$ pode ser escrito em termos deles:

$$
\vec{R} = \alpha \vec{A} + \beta \vec{B} + \gamma \vec{C}
$$

Formando o produto escalar desta equação com $\vec{B} \times \vec{C}$, obtemos então

$$
\vec{R} \cdot (\vec{B} \times \vec{C}) = \alpha \vec{A} \cdot (\vec{B} \times \vec{C})
$$

portanto,

$$
\alpha = \frac{\vec{R} \cdot (\vec{B} \times \vec{C})}{\vec{A} \cdot (\vec{B} \times \vec{C})}
$$

Expressões análogas podem ser escritas para $\beta$ e $\gamma$. Os parâmetros $\alpha$, $\beta$, e $\gamma$ são unicamente determinados, desde que:

$$ \vec{A} \cdot (\vec{B} \times \vec{C}) \neq 0 $$

ou seja, desde que os três vetores base não sejam coplanares.

## Usando a Álgebra Vetorial no Eletromagnetismo

Em um mundo onde a ciência se entrelaça com a arte, a álgebra vetorial se ergue como uma ponte sólida entre o visível e o invisível. Neste ponto da nossa jornada, navegaremos pelas correntes do eletromagnetismo, uma jornada onde cada vetor conta uma história, cada Produto Escalar revela uma conexão profunda, e cada Produto Vetorial desvenda um mistério. A matemática da Álgebra Vetorial é a ferramenta que nos guiará.

Prepare-se para uma surpresa olhe com cuidado e verá como a matemática se torna poesia, desvendando os segredos do universo elétrico e magnético. Esta rota promete uma jornada de descoberta, compreensão e surpresa. Começaremos pelo mais básico de todos os básicos, a Lei de Coulomb.

### Lei de Coulomb

No ano da glória de 1785, um pesquisador francês, [Charles-Augustin de Coulomb](https://en.wikipedia.org/wiki/Charles-Augustin_de_Coulomb)Formulou, empiricamente uma lei para definir a intensidade da força exercida por uma carga elétrica $Q$ sobre outra dada por:

$$
F_{21} = K_e \frac{Q_1Q_2}{R^2}
$$

[Charles-Augustin de Coulomb](https://en.wikipedia.org/wiki/Charles-Augustin_de_Coulomb) estabeleceu sua lei de forma empírica utilizando uma balança de torção para medir as forças de interação entre cargas elétricas estacionárias. Utilizando este método, ele foi capaz de quantificar a relação inversa entre a força e o quadrado da distância entre as cargas. De forma independente, [Henry Cavendish](https://en.wikipedia.org/wiki/Henry_Cavendish) chegou à mesma equação anos depois, também utilizando uma balança de torção, embora seus resultados não tenham sido amplamente publicados na época.
  
Até o surgimento do trabalho de [Michael Faraday](https://en.wikipedia.org/wiki/Michael_Faraday) sobre linhas de força elétrica, a equação desenvolvida por Coulomb era considerada suficiente para descrever interações eletrostáticas. Quase um século depois de Coulomb, matemáticos como [Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss), [Hamilton](https://en.wikipedia.org/wiki/William_Rowan_Hamilton), [Maxwell](https://en.wikipedia.org/wiki/James_Clerk_Maxwell) reformularam esta lei, incorporando-a em um contexto vetorial. Eles utilizaram o cálculo vetorial para expressar as direções e magnitudes da força, permitindo que Lei de Coulomb possa ser aplicada de forma mais geral em campos eletrostáticos e magnetostáticos.

$$
F_{21} = \frac{1}{4\pi \epsilon_0 \epsilon_r} \frac{Q_1Q_2}{R^2} a_{21}
$$

Nesta equação:

- $F_{21}$ é a força que é aplicada sobre a carga 2, $Q_2$, devido a existência da carga 1, $Q_1$.
- $\epsilon_0$ representa a permissividade do vácuo, medida em Farads por metro ($F/m$).
- $\epsilon_r$ representa a permissividade do meio onde as cargas estão, um valor escalar e sem unidade.
- $4\pi $ surge da existência da força em todos os pontos do espaço, uma esfera que se estende da carga até o infinito.
- $Q_1Q_2$ representa o produto entre as intensidades das cargas que no Sistema Internacional de Unidades são medidas em Coulombs ($C$).
- $a_{21}$ representa o vetor unitário com origem em $Q1$ e destino em $Q2$.

## Cálculo Vetorial

Cálculo vetorial, soa como algo saído de uma história de ficção científica. Mas é mais terra-a-terra do que podemos imaginar de longe. Trata-se uma técnica para lidar com quantidades que têm tanto magnitude quanto direção de forma contínua. Velocidade. Força. Fluxo de um rio, Campos Elétricos, Campos Magnéticos. Coisas que não apenas têm um tamanho, mas também uma direção, um sentido. Não sei se já falei sobre isso, são as grandezas que chamamos de vetoriais e representamos por vetores.

A beleza do cálculo vetorial perceptível na sua capacidade de descrever o mundo físico profunda e significativamente. 

Considere um campo de trigo balançando ao vento. O vento não está apenas soprando com uma certa força, mas também em uma certa direção. O cálculo vetorial nos permitirá entender fenômenos como esse e transformá-los em ferramentas de inovação e sucesso.

O cálculo vetorial é construído sobre três operações fundamentais: o gradiente, a divergência e o rotacional. O gradiente nos diz a direção e a taxa na qual uma quantidade está mudando. A divergência nos diz o quanto um campo está se espalhando de um ponto. E o rotacional nos dá uma medida da rotação ou vorticidade de um campo.

Se tivermos uma função escalar $\mathbf{F}$, o gradiente de $\mathbf{F}$ será dado por:

$$
\nabla \mathbf{F} = \left( \frac{\partial \mathbf{F}}{\partial x}, \frac{\partial \mathbf{F}}{\partial y}, \frac{\partial \mathbf{F}}{\partial z} \right)
$$

Se tivermos um campo vetorial $ \mathbf{F} = F_x \, \vec{a}_x + F_y \, \vec{a}_y + F_z \, \vec{a}_x $, a divergência de $\mathbf{F}$ é dada por:

$$
\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
$$

O rotacional de $\mathbf{F}$ será dado por:

$$
\nabla \times \mathbf{F} = \left( \frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z} \right) \, \vec{a}_x - \left( \frac{\partial F_z}{\partial x} - \frac{\partial F_x}{\partial z} \right) a_i + \left( \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y} \right) \, \vec{a}_z
$$

A única coisa que pode encher seus olhos de lágrimas é o sal trazido pela maresia, não o medo do Cálculo Vetorial. Então, não se intimide por estas equações herméticas, quase esotéricas. O Cálculo Vetorial é apenas conjunto de ferramentas, como um canivete suíço, que nos ajuda a explorar e entender o mundo ao nosso redor. Nós vamos abrir cada ferramenta deste canivete e aprender a usá-las.

### Campos Vetoriais

Quando olhamos as grandezas escalares, traçamos Campos Escalares. Como uma planície aberta, eles se estendem no espaço, sem direção, mas com magnitude, definidos por uma função $\mathbf{F}(x,y,z)$, onde $x$, $y$, $z$ pertencem a um universo de triplas de números reais. Agora, para as grandezas vetoriais, moldamos Campos Vetoriais, definidos por funções vetoriais $\mathbf{F}(x,y,z)$, onde $x$, $y$, $z$ são componentes vetoriais. Em outras palavras, representamos Campos Vetoriais no espaço como um sistema onde cada ponto do espaço puxa um vetor.

Imagine-se em um rio, a correnteza o arrastando, conduzindo seu corpo. A correnteza aplica uma força sobre seu corpo. O rio tem uma velocidade, uma direção. Em cada ponto, ele te empurra de uma forma diferente. Isso é um campo vetorial. Ele é como um mapa, com forças distribuídas, representadas por setas desenhadas para te orientar. Mas essas setas não são meras orientações. Elas têm um comprimento, uma magnitude, e uma direção e um sentido. Elas são vetores. E o mapa completo, deste rio com todas as suas setas, descreverá um campo vetorial.

Em cada ponto no espaço, o campo vetorial tem um vetor. Os vetores podem variar de ponto para ponto. Pense de novo no rio. Em alguns lugares, a correnteza é forte e rápida. Em outros, é lenta e suave. Cada vetor representará essa correnteza em um ponto específico. E o campo vetorial representará o rio todo.

Frequentemente, Campos Vetoriais são chamados para representar cenas do mundo físico: a ação das forças na mecânica, o desempenho dos campos elétricos e magnéticos no Eletromagnetismo, o fluxo de fluidos na dinâmica dos fluidos. Em cada ponto, as coordenadas $(x, y, z)$ são protagonistas, ao lado das funções escalares $P$, $Q$ e $R$. O vetor resultante no palco tem componentes nas direções $x$, $y$ e $z$, representadas pelos atores coadjuvantes, os vetores unitários $(\, \vec{a}_x, \, \vec{a}_y, \, \vec{a}_z)$.

Imaginar um campo vetorial no palco do espaço tridimensional é tarefa árdua que requer visão espacial, coisa para poucos. Para aqueles que já trilharam os caminhos árduos da geometria e do desenho tridimensional Se nosso palco for bidimensional, poderemos colocar os vetores em um plano, selecionar alguns pontos e traçar estes vetores. Neste caso voltaremos nossa atenção e esforço para trabalhar com apenas os componentes $x$ e $y$ e o campo vetorial será definido por uma função dada por:

$$\mathbf{F}(x, y) = (P(x, y), Q(x, y))$$

Uma função, uma definição direta, e simples, ainda assim, sem nenhum apelo visual. Mas somos insistentes e estamos estudando matemática, a rota que nos levará ao horizonte do Eletromagnetismo. Que nasce na carga elétrica, fenômeno simples, estrutural e belo que cria forças que se espalham por todo universo. Vamos pegar duas cargas de mesma intensidade e colocar no nosso palco.

![Campo Vetorial devido a duas cargas elétricas](/assets/images/CampoVetorial1.jpeg){:class="lazyimg"}

<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Figura 3 - Diagrama de um campo vetorial em duas dimensões.</legend>

Agora podemos ver o Campo Vetorial, simples, com poucos pontos escolhidos no espaço e duas cargas pontuais representadas por círculos. Um vermelho, quente, para indicar a carga positiva outro azul, frio, para indicar a carga negativa. Treine a vista. Seja cuidadoso, detalhista. E verá a interação das forças em todos os pontos do espaço.

O Campo Elétrico, o Campo Vetorial que a figura apresenta, surge, na força da própria definição, na carga elétrica positiva por isso os vetores apontam para fora, para longe desta carga, divergem. E são drenados pela carga elétrica negativa, as setas apontam diretamente para ela, convergem. Em todos os pontos que escolhi para plotar em todo o espaço do plano desenhado, você pode ver o efeito das forças criadas por esta carga. Em alguns pontos um vetor está exatamente sobre o outro, eles se anulam, em todos os outros pontos do espaço se somam.

Visualizar um Campo Vetorial é como assistir a uma peça, com cada vetor como um ator em um gráfico. Cada vetor é um personagem desenhado com uma linha direcionada, geralmente com uma seta, atuando com direção e magnitude. Mas essa peça é complexa e exige tempo e paciência para ser compreendida. Uma abordagem mais simples seria tomar um ponto de teste no espaço e desenhar algumas linhas entre a origem do Campo Vetorial e esse ponto, traçando assim os principais pontos da trama.

O Campo Vetorial requer cuidado, carinho e atenção, ele está em todos os pontos do espaço. Contínuo e muitas vezes, infinito. Trabalhar com a continuidade e com o infinito requer mãos calejadas e fortes. Teremos que recorrer a Newton e [Leibniz](https://en.wikipedia.org/wiki/Gottfried_Wilhelm_Leibniz) e ao Cálculo Integral e Diferencial. Não tema! Ainda que muitos se acovardem frente a continuidade este não será nosso destino. Vamos conquistar integrais e diferenciais como [Odisseu](https://en.wikipedia.org/wiki/Odysseus) conquistou [Troia](https://en.wikipedia.org/wiki/Trojan_War), antes de entrar em batalha vamos afiar espadas, lustrar escudos e lanças, na forma de gradiente, divergência e rotacional.

### Gradiente

Imagine-se no topo de uma montanha, cercado por terreno acidentado. Seu objetivo é descer a montanha, mas o caminho não é claramente marcado. Você olha ao redor, tentando decidir para qual direção deve seguir. O gradiente é como uma bússola que indica a direção de maior inclinação. Se você seguir o gradiente, estará se movendo na direção de maior declividade. Se a velocidade for importante é nesta direção que descerá mais rápido.

Agora, vamos trazer um pouco de matemática para esta metáfora. Em um espaço de múltiplas dimensões. Imagine uma montanha com muitos picos e vales, e você pode se mover em qualquer direção, o gradiente de uma função em um determinado ponto é um vetor que aponta na direção de maior variação desta função. Se a função tem múltiplas dimensões, **o gradiente é o vetor que resulta da aplicação das derivadas parciais da função**.

Se tivermos uma função $\mathbf{F}(x, y)$, uma função escalar, o gradiente de $\mathbf{F}$ será dado por:

$$
\nabla \mathbf{F} = \left( \frac{\partial \mathbf{F}}{\partial x}, \frac{\partial \mathbf{F}}{\partial y} \right)
$$

Assim como a bússola na montanha, o gradiente nos mostra a direção à seguir para maximizar, ou minimizar, a função. É uma ferramenta importante na matemática e na física, especialmente em otimização e aprendizado de máquina. Mas não tire seus olhos do ponto mais importante: **o gradiente é uma operação que aplicada a uma função escalar devolve um vetor**. Em três dimensões, usando o Sistema de Coordenadas Cartesianas teremos:

$$
\nabla \mathbf{F} = \left( \frac{\partial \mathbf{F}}{\partial x}, \frac{\partial \mathbf{F}}{\partial y}, \frac{\partial \mathbf{F}}{\partial z} \right)
$$

onde $\frac{\partial \mathbf{F} }{\partial x}$, $\frac{\partial \mathbf{F}}{\partial y}$, e $\frac{\partial \mathbf{F}}{\partial z}$ são as derivadas parciais de $\mathbf{F}$ com respeito a $x$, $y$, e $z$ respectivamente.

Só a expressão **Derivadas parciais** pode fazer o coração bater mais rápido. O medo não o guiará aqui. As derivadas parciais são como velhos amigos que você ainda não conheceu.

Imagine-se em uma grande pradaria. O vento está soprando, carregando consigo o cheiro da grama e da terra. Você está livre para caminhar em qualquer direção. Para o norte, onde o sol se põe, ou para o sul, onde a floresta começa. Cada passo que você dá muda a paisagem ao seu redor, mas de maneiras diferentes dependendo da direção em que você escolheu caminhar.

A derivada parcial é apenas essa ideia, vestida com a roupa do cálculo. Ela apenas quer saber: e se eu der um pequeno passo para o norte, ou seja, mudar um pouco $x$, como a paisagem, nossa função, vai mudar? Ou o se for para o sul, ou em qualquer outra direção que escolher.

Então, em vez de temer as derivadas parciais, podemos vê-las como uma ferramentas úteis que nos ajudem a entender a terra sob nossos pés, o vento, a água que flui, o Campo Elétrico, entender a função que estamos usando para descrever o fenômeno que queremos entender. Com as derivadas parciais, podemos entender melhor o terreno onde pisamos, saber para onde estamos indo e como chegar lá. E isso é bom. Não é?

Uma derivada parcial de uma função de várias variáveis revela a taxa na qual a função muda quando pequenas alterações são feitas em apenas uma das incógnitas da função, mantendo todas as outras constantes. O conceito é semelhante ao conceito de derivada em cálculo de uma variável, entretanto agora estamos considerando funções com mais de uma incógnita.

Por exemplo, se temos uma função $\mathbf{F}(x, y)$, a derivada parcial de $\mathbf{F}$ em relação a $x$ (denotada por $\frac{\partial \mathbf{F}}{\partial x}$ mede a taxa de variação de $\mathbf{F}$ em relação a pequenas mudanças em $x$, mantendo $y$ constante. Da mesma forma, $\frac{\partial \mathbf{F}}{\partial y}$ mede a taxa de variação de $\mathbf{F}$ em relação a pequenas mudanças em $y$, mantendo $x$ constante. Em três dimensões, a derivada parcial em relação uma das dimensões é a derivada de $\mathbf{F}$ enquanto mantemos as outras constantes. Nada mais que a repetição, dimensão a dimensão da derivada em relação a uma dimensão enquanto as outras são constantes.

**O gradiente mede a taxa em que o Campo Escalar varia em uma determinada direção.** Para clarear e afastar a sombra das dúvidas, nada melhor que um exemplo.

<p class="exp">
<b>Exemplo 13:</b><br>
Considerando o Campo Escalar dado por $\mathbf{F}(x,y) = 10sin(\frac{x^2}{5})+4y$, (a) calcule a intensidade do campo no ponto $P(2,3)$, (b) o gradiente deste campo no ponto $P$.  
<br><br>
<b>Solução:</b><br>

(a) A intensidade em um ponto é trivial, trata-se apenas da aplicação das coordenadas do ponto desejado na função do campo. Sendo assim:

\[\mathbf{F}(x,y) = 10sin(\frac{x^2}{5})+4y\]

\[\mathbf{F}(2,3) = 10sin(\frac{2^2}{5})\, \vec{a}_x+4(3)\, \vec{a}_y\]

\[\mathbf{F}(2,3) = 7.17356\, \vec{a}_x+12\, \vec{a}_y\]

(b) agora precisamos calcular o gradiente. O gradiente de uma função $\mathbf{F}(x, y)$ é um vetor que consiste nas derivadas parciais da função com respeito a cada uma de suas variáveis que representam suas coordenadas. <br><br>

Vamos calcular as derivadas parciais de $\mathbf{F}$ com respeito a $x$ e $y$, passo a passo:<br><br>

Primeiro, a derivada parcial de $f$ com respeito a $x$ é dada por:

\[
\frac{\partial \mathbf{F}}{\partial x} = \frac{\partial}{\partial x} \left[10\sin\left(\frac{x^2}{5}\right) + 4y\right]
\]

Nós podemos dividir a expressão em duas partes e calcular a derivada de cada uma delas separadamente. A derivada de uma constante é zero, então a derivada de $4y$ com respeito a $x$ é zero. Agora, vamos calcular a derivada do primeiro termo:

\[
\frac{\partial}{\partial x} \left[10\sin\left(\frac{x^2}{5}\right)\right] = 10\cos\left(\frac{x^2}{5}\right) \cdot \frac{\partial}{\partial x} \left[\frac{x^2}{5}\right]
\]

Usando a regra da cadeia, obtemos:

\[
10\cos\left(\frac{x^2}{5}\right) \cdot \frac{2x}{5} = \frac{20x}{5}\cos\left(\frac{x^2}{5}\right) = 4x\cos\left(\frac{x^2}{5}\right)
\]

Portanto, a derivada parcial de $\mathbf{F}$ com respeito a $x$ é:

\[
\frac{\partial \mathbf{F}}{\partial x} = 4x\cos\left(\frac{x^2}{5}\right)
\]

Agora, vamos calcular a derivada parcial de $\mathbf{F}$ com respeito a $y$:

\[
\frac{\partial \mathbf{F}}{\partial y} = \frac{\partial}{\partial y} \left[10\sin\left(\frac{x^2}{5}\right) + 4y\right]
\]

Novamente, dividindo a expressão em duas partes, a derivada do primeiro termo com respeito a $y$ é zero (pois não há $y$ no termo), e a derivada do segundo termo é $4$. Portanto, a derivada parcial de $\mathbf{F}$ com respeito a $y$ é:

\[
\frac{\partial \mathbf{F}}{\partial y} = 4
\]

Assim, o gradiente de $\mathbf{F}$ é dado por:

\[
\nabla \mathbf{F} = \left[\frac{\partial \mathbf{F}}{\partial x}, \frac{\partial \mathbf{F}}{\partial y}\right] = \left(4x\cos\left(\frac{x^2}{5}\right), 4\right)
\]

E esta é a equação que define o gradiente. Para saber o valor do gradiente no ponto $P$ tudo que precisamos é aplicar o ponto na equação então:

\[
\nabla \mathbf{F}(2,3) = \left( 4(2)\cos\left(\frac{2^2}{5}\right), 4\right) = \left(  5.57365, 4 \right)
\]

Ao derivarmos parcialmente o Campo Vetorial $\mathbf{F}$ escolhemos nosso Sistema de Coordenadas. Sendo assim:

\[
\nabla \mathbf{F}(2,3) = 5.57365 \, \vec{a}_x+ 4\, \vec{a}_y
\]
</p>

Assim como um navegador considera a variação da profundidade do oceano em diferentes direções para traçar a rota mais segura, a derivada parcial nos ajuda a entender como uma função se comporta quando mudamos suas variáveis de entrada. O gradiente é a forma de fazermos isso em todas as dimensões, derivando em uma incógnita de cada vez.

#### Significado do Gradiente

Em qualquer ponto $P$ o gradiente é um vetor que aponta na direção da maior variação de um Campo Escalar neste ponto. Nós podemos voltar ao exemplo 8 e tentar apresentar isso de uma forma mais didática. Primeiro o gráfico do Campo Escalar dado por: $\mathbf{F}(x,y) = 10sin(\frac{x^2}{5})+4y$.

![Gráfico do Campo Escalar](/assets/images/Func1Grad.jpeg){:# class="lazyimg"}
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Figura 4 - Gráfico de um Campo Escalar $f(x,y)$.</legend>

Na Figura 4 é possível ver a variação do do campo $\mathbf{F}(x,y)$ eu escolhi uma função em $\mathbf{F}(x,y)$ no domínio dos $\mathbb{R}^2$ por ser mais fácil de desenhar e visualizar, toda a variação fica no domínio de $z$. Podemos plotar o gradiente na superfície criada pelo campo $\mathbf{F}(x,y)$.

![Gráfico do Campo Escalar mostrando a intensidade do gradiente ](/assets/images/func1Grad2.jpeg){:class="lazyimg"}
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Figura 5 - Gráfico de um Campo Escalar $f(x,y) representando o Gradiente$.</legend>

Em cada ponto da Figura 5 a cor da superfície foi definida de acordo com a intensidade do gradiente. Quanto menor esta intensidade, mais próximo do vermelho. Quanto maior, mais próximo do Azul. Veja que a variação é maior nas bordas de descida ou subida e menor nos picos e vales. Coisas características da derivação.

É só isso. Se a paciente leitora entendeu até aqui, entendeu o gradiente e já sabe aplicá-lo. Eu disse que a pouparia de lágrimas desnecessárias. E assim o fiz.

#### Propriedades do Gradiente

O reino do gradiente é o reino dos Campos Escalares, o gradiente tem características matemáticas distintas que o guiam em sua exploração:

1. **Linearidade**: O gradiente é uma operação linear. Isso significa que para quaisquer campos escalares $f$ e $g$, e quaisquer constantes $a$ e $b$, temos:

    $$
      \nabla (af + bg) = a \nabla f + b \nabla g
    $$

    O gradiente de uma soma de funções é a soma dos gradientes das funções, cada um ponderado por sua respectiva constante.

2. **Produto por Escalar**: O gradiente de uma função escalar multiplicada por uma constante é a constante vezes o gradiente da função. Para uma função escalar $f$ e uma constante $a$, teremos:

    $$
    \nabla (af) = a \nabla f
    $$

3. **Regra do Produto**: Para o produto de duas funções escalares $f$ e $g$, a regra do produto para o gradiente é dada por:

    $$
    \nabla (fg) = f \nabla g + g \nabla f
    $$

    Esta é a versão para gradientes da regra do produto para derivadas no cálculo unidimensional.

4. **Regra da Cadeia**: Para a função composta $f(g(x))$, a regra da cadeia para o gradiente será dada por:

    $$
    \nabla f(g(x)) = (\nabla g(x)) f'(g(x))
    $$

    Esta é a extensão da regra da cadeia familiar do cálculo unidimensional.

Estas propriedades, como as leis imutáveis da física, regem a conduta do gradiente em sua jornada através dos campos escalares. No palco do eletromagnetismo, o gradiente desempenha um papel crucial na descrição de como os Campos Elétrico e Magnético variam no espaço.

1. **Campo Elétrico e Potencial Elétrico**: o campo elétrico é o gradiente negativo do potencial elétrico. Isso significa que o Campo Elétrico aponta na direção de maior variação do potencial elétrico, formalmente expresso como:

    $$
    \mathbf{E} = -\nabla V
    $$

    Aqui, $\mathbf{E}$ é o Campo Elétrico e $V$ é o potencial elétrico. O gradiente, portanto, indica a encosta, o aclive, mais íngreme que uma partícula carregada experimentaria ao mover-se no Campo Elétrico.

2. **Campo Magnético**: o Campo Magnético não é o gradiente de nenhum potencial escalar, **O Campo Magnético é um campo vetorial cuja divergência é zero**. No entanto, em situações estáticas ou de baixas frequências, pode-se definir um potencial vetorial $\mathbf{A}$ tal que:

    $$ \mathbf{B} = \nabla \times \mathbf{A} $$

Essas propriedades do gradiente são como setas, apontando o caminho através das complexidades do eletromagnetismo. O gradiente é a ferramenta mais simples do nosso canivete suíço do cálculo vetorial.

### Divergência

Seu barco sempre será pequeno perto do oceano e da força do vento. Você sente o vento em seu rosto, cada sopro, uma força direcional, um vetor com magnitude e direção. Todo o oceano e a atmosfera acima dele compõem um campo vetorial, com o vento soprando em várias direções, forças aplicadas sobre o seu barco.

No oceano, o tempo é um caprichoso mestre de marionetes, manipulando o clima com uma rapidez alucinante. Agora, em uma tempestade, existem lugares onde o vento parece convergir, como se estivesse sendo sugado para dentro. Em outros lugares, parece que o vento está explodindo para fora. Esses são os pontos de divergência e convergência do campo vetorial do vento.

Um lugar onde o vento está sendo sugado para dentro tem uma divergência negativa - o vento está "fluido" para dentro mais do que está saindo. Um lugar onde o vento está explodindo para fora tem uma divergência positiva - o vento está saindo mais do que está entrando. Este é o conceito que aplicamos as cargas elétricas, o campo elétrico diverge das cargas positivas e converge para as negativas. Isto porque assim convencionamos há séculos, quando começamos e estudar o Eletromagnetismo.

Matematicamente, **a divergência é uma maneira de medir esses comportamentos de "expansão" ou "contração" de um campo vetorial**. Para um campo vetorial em três dimensões, $\mathbf{F} = f_x \, \vec{a}_x + f_y \, \vec{a}_y + f_z \, \vec{a}_z$, a divergência é calculada como:

$$
\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
$$

A divergência, então, é como a "taxa de expansão" do vento em um determinado ponto - mostra se há mais vento saindo ou entrando em uma região específica do espaço, um lugar. assim como a sensação que temos no meio de uma tempestade. **A divergência é o resultado do Produto Escalar entre o operador $\nabla$ e o Campo Vetorial. O resultado da divergência é uma função escalar que dá a taxa na qual o fluxo do campo vetorial está se expandindo ou contraindo em um determinado ponto**.

Sendo um pouco mais frio podemos dizer que a divergência é um operador diferencial que atua sobre um Campo Vetorial para produzir um Campo Escalar. Em termos físicos, a divergência em um ponto específico de um Campo Vetorial representa a fonte ou dreno no ponto: uma divergência positiva indica que neste ponto existe uma fonte, ou fluxo de vetores para fora, divergindo. Enquanto uma divergência negativa indica um dreno ou fluxo para dentro, convergindo.

#### Fluxo e a Lei de Gauss

O fluxo, nas margens do cálculo vetorial, é **uma medida da quantidade de campo que passa através de uma superfície**. Imagine um rio, com a água fluindo com velocidades e direções variadas. Cada molécula de água tem uma velocidade - um vetor - e toda a massa de água compõe um Campo Vetorial.

Se você colocar uma rede no rio, o fluxo do campo de água através da rede seria uma medida de quanta água está passando por ela. Para um campo vetorial $\mathbf{F}$ e uma superfície $S$ com vetor normal dado por $\mathbf{n}$, o fluxo será definido, com a formalidade da matemática, como:

$$
\iint_S (\mathbf{F} \cdot \mathbf{n}) \, dS
$$

Uma integral dupla, integral de superfície onde $dS$ é o elemento diferencial de área da superfície, e o Produto Escalar $\mathbf{F} \cdot \mathbf{n}$ mede o quanto do campo está fluindo perpendicularmente à superfície.

Agora, a divergência entra em cena como a versão local do fluxo. Se encolhermos a rede até que ela seja infinitesimalmente pequena, o fluxo através da rede se tornará infinitesimal e será dado pela divergência do campo de água no ponto onde a rede está. Matematicamente, isso é expresso na Lei de [Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss):

$$
\nabla \cdot \mathbf{F} = \frac{d (\text{Fluxo})}{dV}
$$

Onde, $V$ é o volume da região, e **a Lei de Gauss afirma que a divergência de um campo vetorial em um ponto é igual à taxa de variação do fluxo do campo através de uma superfície que envolve o ponto**.

#### Teorema da Divergência

Imagine-se como um explorador atravessando o vasto terreno do cálculo vetorial. Você se depara com duas paisagens: a superfície e o volume. Cada uma tem suas próprias características e dificuldades, mas há uma ponte que as conecta, uma rota que permite viajar entre elas. Esta é a Lei de Gauss.

A Lei de Gauss, ou o Teorema da Divergência, é a ponte que interliga dois mundos diferentes. Ela afirma que, para um dado campo vetorial $\mathbf{F}$, a integral de volume da divergência do campo vetorial sobre um volume $V$ é igual à integral de superfície do campo vetorial através da superfície $S$ que delimita o volume $V$:

$$
\iiint_V (\nabla \cdot \mathbf{F}) \, dV = \iint_S (\mathbf{F} \cdot \mathbf{n}) \, dS
$$

Uma integral tripla igual a uma integral dupla. Aqui, $dV$ é um pedaço infinitesimalmente pequeno de volume dentro de $V$, e $dS$ é um pedaço infinitesimalmente pequeno da superfície $S$, respectivamente elemento infinitesimal de volume e área. O vetor $\mathbf{n}$ é um vetor normal apontando para fora da superfície.

Com a Lei de Gauss, podemos ir e voltar entre a superfície e o volume, entre o plano e o volume. Esta é a beleza e o poder da matemática: a linguagem e as ferramentas para navegar pelos mais complexos terrenos.

#### Propriedades da Divergência

No universo dos campos vetoriais, a divergência tem propriedades matemáticas distintas que servem como marcos na paisagem:

1. **Linearidade**: A divergência é uma operação linear. Isso significa que para quaisquer campos vetoriais $\mathbf{F}$ e $mathbf{G}$, e quaisquer escalares $a$ e $b$, temos:

    $$
        \nabla \cdot (a\mathbf{F} + b\mathbf{G}) = a (\nabla \cdot \mathbf{F}) + b (\nabla \cdot \mathbf{G})
    $$

    A divergência de uma soma é a soma das divergências, com cada divergência ponderada por seu respectivo escalar.

2. **Produto por Escalar**: A divergência de um campo vetorial multiplicado por um escalar é o escalar vezes a divergência do campo vetorial. Para um campo vetorial $\mathbf{F}$ e um escalar $a$, temos:

    $$
        \nabla \cdot (a\mathbf{F}) = a (\nabla \cdot \mathbf{F})
    $$

3. **Divergência de um Produto**: A divergência de um produto de um campo escalar $\phi$ e um campo vetorial $\mathbf{F}$ é dado por:

    $$
    \nabla \cdot (\phi \mathbf{F}) = \phi (\nabla \cdot \mathbf{F}) + \mathbf{F} \cdot (\nabla \phi)
    $$

Este é o análogo vetorial do produto de regra para derivadas no cálculo unidimensional.

4. **Divergência do Rotação**: A divergência do rotacional de qualquer campo vetorial é sempre zero:

    $$
      \nabla \cdot (\nabla \times \mathbf{F}) = 0
    $$

Esta propriedade é um reflexo do fato de que as linhas de campo do rotacional de um campo vetorial são sempre fechadas, sem início ou fim. Não se preocupe, ainda, já, já, chegaremos ao rotacional.

Essas propriedades são como as leis inabaláveis que governam o comportamento da divergência em sua jornada através dos campos vetoriais.

### Rotacional

Imagine estar no meio de um tornado, onde o vento gira em um padrão circular em torno de um ponto central. O movimento deste vento pode ser descrito como um campo vetorial, que tem tanto uma direção quanto uma magnitude em cada ponto no espaço. Agora, considere um pequeno ponto neste campo - o rotacional é uma operação matemática que lhe dirá quão rapidamente e em que direção o vento está girando em torno deste ponto.

Para entender isso, vamos recorrer à matemática. **O rotacional é um operador diferencial que atua sobre um campo vetorial, produzindo outro campo vetorial que descreve a rotação local do campo original**. Se considerarmos um campo vetorial em três dimensões, representado por $\mathbf{F}(x, y, z)$, o rotacional desse campo será dado por:

$$
\nabla \times \mathbf{F} = \left(\frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}\right)\mathbf{i} + \left(\frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}\right)\mathbf{j} + \left(\frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}\right)\mathbf{k}
$$

Esta operação fornece uma descrição da rotação em cada ponto no espaço, sendo um vetor perpendicular ao plano de rotação, cuja magnitude representa a velocidade de rotação. Em outras palavras, **o rotacional de um campo vetorial em um ponto particular indica quão _giratório_ é o campo naquele ponto**.

Imagine agora que você está no meio de um rio, onde a água gira em torno de algumas pedras, criando redemoinhos. O rotacional, nesse caso, poderia descrever como a água gira em torno desses pontos, permitindo-nos entender a dinâmica do fluxo de água neste rio.

É como uma dança, onde cada ponto no espaço executa uma rotação única, formando uma coreografia complexa e bela. Esta coreografia é o campo vetorial em questão. Entender o rotacional permite desvendar os segredos por trás dos padrões de fluxo em campos vetoriais, sejam eles campos de vento, campos magnéticos ou correntes de água.

No contexto do eletromagnetismo, o rotacional tem uma aplicação crucial, especialmente ao considerarmos as Equações de Maxwell. Uma das equações de Maxwell, especificamente a Lei de Ampère-Maxwell, é expressa como

$$
\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}
$$

onde $\nabla \times \mathbf{B}$ é o rotacional do campo magnético $\mathbf{B}$, $\mu_0$ é a permeabilidade do vácuo, $\mathbf{J}$ é a densidade de corrente elétrica e $\frac{\partial \mathbf{E}}{\partial t}$ é a variação do campo elétrico com relação ao tempo.

A Lei de Ampére-Maxwell representa a relação entre a corrente elétrica variável no tempo e o campo magnético rotativo que é gerado, sendo uma descrição matemática do fenômeno da indução eletromagnética. Desta forma, a operação do rotacional serve como uma ponte para unir e descrever fenômenos eletromagnéticos interdependentes, facilitando a análise e compreensão das interações complexas entre campos elétricos e magnéticos, essencial para a física moderna e inovações tecnológicas.

___
[^1]: SADIKU, Matthew N.O., Elementos de Eletromagnetismo. Porto Alegre: Bookman, 5a Edição, 2012  
[^2]: VENTURE, Jair J.. Álgebra Vetorial e Geometria Analítica. Curitiba - PR: Livrarias Curitiba, 10a Edição, 2015
