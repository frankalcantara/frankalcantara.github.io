module Main where
import Coisa
main :: IO ()

far :: Float -> Float
far c = 32+c/5*9

{-
  quero uma fun��o que me devolva um pre�o segundo as regras a 
  seguir: 
  1. se pre�o for menor que 10 devolva pre�o mais 2;
  2. se pre�o for menor que 20 devolva pre�o mais 1;
  3. se pre�o for maior que 20 devolva pre�o mais 0.5.
-}
preco :: Double -> Double
preco p
  |p < 10 = p+2
  |p < 20 = p+1
  |otherwise = p+0.5

  {- 
    fazer uma fun��o que retorne, a partir de um inteiro digitado,
    a lista de todos os inteiros recursivamente at� um seguindo
    a seguinte regra: se o n�mero digitado for divis�vel por dois
    o resultado desta divis�o inteira por dois; se o n�mero digitado n�o for divisivel por dois retornamos 3*numero+1. 
    Este processo para quando o resultado for 1
    Por exemplo, o usu�rio digita 10
    [10, 5, 16, 8, 4, 2, 1]
  -}

next :: Int -> Int
next n
  |n `mod` 2 == 0 = n `div` 2
  |otherwise = 3*n+1

numeros :: Int -> [Int]  
numeros n
  |n == 1 = [1]
  |otherwise = n : numeros (next n)

main = do
  print $ dobra 2
  print (far 32)
  print (preco 5)
  print (preco 12)
  print (preco 22)

{-
  print "Digite Qualquer coisa: "
  qualquerCoisa <- getLine
  
  print ("Voce escreveu " ++ qualquerCoisa)
  print $ "Voce escreveu " ++ qualquerCoisa

  print "Digite um numero inteiro"
 
  numero <- getLine
  print (far (read numero))

  print (5 `mod` 2) {- retorna o resto de uma divis�o de inteiros-}
  print (5 `div` 2) {- retorna o quociente da divis�o de inteiros-}
-} 
  print (numeros 57)

{-
  Fa�a com o que aprendeu at� o momento, uma calculadora, n�meros reais, para as quatro opera��es acrescida das fun��es raiz quadrada, seno e cosseno (radianos). 
-}  
 




