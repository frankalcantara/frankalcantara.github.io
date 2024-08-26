module Main where
--função para cálculo do fatorial
-- fatorial :: Int->Int
-- fatorial 0 = 1
-- fatorial n = n * fatorial (n-1)

-- dez minutos para você escrever esta função usando guardas e if

--fatorial :: Int->Int
--fatorial n
--  | n == 0 = 1
--  | otherwise = n * fatorial (n-1)

fatorial :: Int->Int
fatorial n = if n == 0 then 1 else n * fatorial (n-1)


{- 
    There are three two-argument exponentiation operations: 
    (^) raises any number to a nonnegative integer power, 
    (^^) raises a fractional number to any integer power, 
    and (**) takes two floating-point arguments. 
    The value of x^0 or x^^0 is 1 for any x, including zero; 0**y is undefined.
-}

{-
    Exemplos com div mod rem e quot
    print (show (2^4)) 
    print (show (4^^(-2))) 
    print (show (2 `div` (-3))) 
    print (show (2 `mod` (-3))) 
    print (show (2 `quot` (-3))) 
    print (show (2 `rem` (-3))) 
-}

{- 
    Cálculo do mdc segundo o algoritmo simplificado de 
    Euclides 
-}
mdc :: Int -> Int -> Int
mdc a b
  | b == 0 = abs a
  | otherwise = mdc b (rem a b) 

{-
    somaDigitos
-}
somaDigitos :: Int -> Int
somaDigitos n 
  | (n > 0 && n <= 9) = n
  | otherwise =
    let esse = n `mod` 10
        resto = n `div` 10
     in esse + somaDigitos resto
{-
Palíndromo 
 -}

palindromo :: String->Bool
palindromo (xs)
  | length xs <= 1 = True
  | head xs /= last xs = False
  | otherwise = 
    palindromo (tail (init xs))

main :: IO ()
main = do
  print (fatorial 4)
  putStrLn (show (fatorial 5))
  print (fatorial 6)