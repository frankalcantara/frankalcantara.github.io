module Main where

{- 
  um modulo em haskell, é uma unidade de compilação que serve a dois propósitos. Controlar namespaces e criar tipos de dados abstratos. 

  Por padrão um programa deve ter pelo menos um módulo Main que exporte a função main.

  -- você pode importar um módulo todo ou apenas algumas funç~ões deste módulo
  import Data.Char (toLower, toUpper) -- import only the functions toLower and toUpper from Data.Char

  import Data.List -- import everything exported from Data.List
 
  import MyModule -- import everything exported from MyModule

  -- ou você pode importar o módulo todo menos algumas funções deste módulo
  import MyModule hiding (remove_e)

  -- Na definição do módulo você pode definir as funções que podem ser exportadas. 

    module MyModule (remove_e, add_two) where

  add_one blah = blah + 1

  remove_e text = filter (/= 'e') text

  add_two blah = add_one . add_one $ blah


-}


{- 
    Em Haskell não existem variáveis, apenas indicadores (nomes) e declarações. 
    Uma vez que um indicador é declarado, ele não pode ser mudado. 
    vamos ver isso no ghci que é mais fácil
  
  
    x = 2000
    x = 1200 

    imagine isso como uma declaração matemática. 
    Mas observe também que isso não é válido se usarmos o let dentro de um do.

    Este é um dos motivos para não termos laços de repetição

  -}

--laço para contar n ocorrências
loop :: Int -> Int
loop n = loop' n 0
  where loop' 0 a = a
        loop' n a = loop' (n - 1) (a + 1) 

f :: Int->Int
f n
  | (n == 0) = 1
  | otherwise = 2 * f (n-1)


{-
  funções de alta ordem, funções que recebem como argumento, outras funções
  print (aplique dobro (take 10 nossa nunNat)
-}  

{-
  Podemos definir funções com tipos polimorficos
-}
primeiro :: (a,b) -> a
primeiro (x,y) = x

  
-- mais testes de funções 
soma1:: Int -> Int
soma1 foo = foo + 1

removeA:: String -> String
removeA texto = filter (/= 'A') texto

-- O que é que isso faz?
--add_two blah = add_one . add_one $ blah


main :: IO ()
main = do
  {- 
    haskell é uma linguagem declarativa, declaramos o que desejamos e não explicitamos
    como isso será feito. O python/C/C++ são imperativas, dizemos como deve ser feito.
  -}

  -- este é um exemplo de avaliação preguiçosa
  let numNat = [1..] -- uma lista de zero até o infinito
  let nossaLista = take 100 numNat
  
  print nossaLista 

  {- 
    Usamos a palavra chave let, sem o in, no corpo de um bloco do.
    Usamos o let também depois da | em compreensão de lista. 
    Em qualquer outro lugar usamos let... in ...

    1. Let - Expression na forma: 
    
    let variavel = expression in Expression

    Esta forma pode ser usado em qualquer lugar onde podemos usar uma expessão

  -}
  print ("Let - Expression: " ++ (show ((let x = 3 in x*2) + 2)))

  -- substituindo o valor 3, let, em x^2 e somando 2 a este resultado.
  {- 
    a função show está declarada no módulo Prelude e tem o tipo: 
    Show a => a -> String
    uma função que recebe um valor a e devolve uma string
  -}

  {- 
    2. Let - declaração na forma: 

    let variável = expression

    Esta é a forma que deve ser usado em um do. Neste caso nunca usamos o in. Por exemplo, como 
    fizemos acima. 

  -}
  {- 
    3. dentro de compreensão de listas, novamente sem o in
      [(x, y) | x <- [1..3], let y = 2*x]  
  -}  
  
  putStrLn "Ate aqui, tudo bem"

  --exemplos dos laços em forma de recursão
  print (loop 10)
  print (f 0)

  -- exemplo de tipos polimorficos
  print (primeiro (1,2))
  print (primeiro ("frank", "Paulo"))

  -- testes de funções 
  print (soma1 20)
  print (removeA "Frank")  