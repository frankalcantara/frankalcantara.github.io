module Main where

{- 
  um modulo em haskell, � uma unidade de compila��o que serve a dois prop�sitos. Controlar namespaces e criar tipos de dados abstratos. 

  Por padr�o um programa deve ter pelo menos um m�dulo Main que exporte a fun��o main.

  -- voc� pode importar um m�dulo todo ou apenas algumas fun�~�es deste m�dulo
  import Data.Char (toLower, toUpper) -- import only the functions toLower and toUpper from Data.Char

  import Data.List -- import everything exported from Data.List
 
  import MyModule -- import everything exported from MyModule

  -- ou voc� pode importar o m�dulo todo menos algumas fun��es deste m�dulo
  import MyModule hiding (remove_e)

  -- Na defini��o do m�dulo voc� pode definir as fun��es que podem ser exportadas. 

    module MyModule (remove_e, add_two) where

  add_one blah = blah + 1

  remove_e text = filter (/= 'e') text

  add_two blah = add_one . add_one $ blah


-}


{- 
    Em Haskell n�o existem vari�veis, apenas indicadores (nomes) e declara��es. 
    Uma vez que um indicador � declarado, ele n�o pode ser mudado. 
    vamos ver isso no ghci que � mais f�cil
  
  
    x = 2000
    x = 1200 

    imagine isso como uma declara��o matem�tica. 
    Mas observe tamb�m que isso n�o � v�lido se usarmos o let dentro de um do.

    Este � um dos motivos para n�o termos la�os de repeti��o

  -}

--la�o para contar n ocorr�ncias
loop :: Int -> Int
loop n = loop' n 0
  where loop' 0 a = a
        loop' n a = loop' (n - 1) (a + 1) 

f :: Int->Int
f n
  | (n == 0) = 1
  | otherwise = 2 * f (n-1)


{-
  fun��es de alta ordem, fun��es que recebem como argumento, outras fun��es
  print (aplique dobro (take 10 nossa nunNat)
-}  

{-
  Podemos definir fun��es com tipos polimorficos
-}
primeiro :: (a,b) -> a
primeiro (x,y) = x

  
-- mais testes de fun��es 
soma1:: Int -> Int
soma1 foo = foo + 1

removeA:: String -> String
removeA texto = filter (/= 'A') texto

-- O que � que isso faz?
--add_two blah = add_one . add_one $ blah


main :: IO ()
main = do
  {- 
    haskell � uma linguagem declarativa, declaramos o que desejamos e n�o explicitamos
    como isso ser� feito. O python/C/C++ s�o imperativas, dizemos como deve ser feito.
  -}

  -- este � um exemplo de avalia��o pregui�osa
  let numNat = [1..] -- uma lista de zero at� o infinito
  let nossaLista = take 100 numNat
  
  print nossaLista 

  {- 
    Usamos a palavra chave let, sem o in, no corpo de um bloco do.
    Usamos o let tamb�m depois da | em compreens�o de lista. 
    Em qualquer outro lugar usamos let... in ...

    1. Let - Expression na forma: 
    
    let variavel = expression in Expression

    Esta forma pode ser usado em qualquer lugar onde podemos usar uma expess�o

  -}
  print ("Let - Expression: " ++ (show ((let x = 3 in x*2) + 2)))

  -- substituindo o valor 3, let, em x^2 e somando 2 a este resultado.
  {- 
    a fun��o show est� declarada no m�dulo Prelude e tem o tipo: 
    Show a => a -> String
    uma fun��o que recebe um valor a e devolve uma string
  -}

  {- 
    2. Let - declara��o na forma: 

    let vari�vel = expression

    Esta � a forma que deve ser usado em um do. Neste caso nunca usamos o in. Por exemplo, como 
    fizemos acima. 

  -}
  {- 
    3. dentro de compreens�o de listas, novamente sem o in
      [(x, y) | x <- [1..3], let y = 2*x]  
  -}  
  
  putStrLn "Ate aqui, tudo bem"

  --exemplos dos la�os em forma de recurs�o
  print (loop 10)
  print (f 0)

  -- exemplo de tipos polimorficos
  print (primeiro (1,2))
  print (primeiro ("frank", "Paulo"))

  -- testes de fun��es 
  print (soma1 20)
  print (removeA "Frank")  