module Main  where

import Prelude hiding (map, filter, foldr, length, foldl, zip, head, tail, span, partition, unzip, zipWith, maybe, (++), concatMap)

main :: IO ()
main = putStrLn "Hello, world!"

--         v construtor
type X = IO
---  ^ sinônimo

data Cor = Red | Green | Blue
  deriving (Eq, Show, Ord, Read, Enum)
  
{-
type Cor :: *
Red   :: Cor
Green :: Cor
Blue  :: Cor
-}

-- data Bool = True | False

not' :: Bool -> Bool
not' x = if x then False else True

not'' :: Bool -> Bool
not'' True = False

not''' :: Bool -> Bool
not''' x =
  case x of
    False -> True

-- if x then T else E 
-- case x of
--   True -> T
--   False -> E

data Nat = Z | S Nat
  deriving (Eq, Ord, Show, Read)

plus :: Nat -> Nat -> Nat
plus Z x = x
plus (S x) y = S (plus x y)

toInt :: Nat -> Int
toInt Z = 0
toInt (S x) = 1 + toInt x

-- 0 + x       = x
-- (1 + k) + n = 1 + (k + n)

-- data [] a = [] | a:[a]

length :: [] a -> Int
length []     = 0
length (a:as) = 1 + length_as where
  length_as = length as

map :: (a -> b) -> [a] -> [b]
map f []     = []
map f (a:li) = f a:map f li

-- map (+ 1) (Cons 1 (Cons 2 (Cons 3 Nil)))
-- = Cons (1 + 1) (map (+1) (Cons 2 (Cons 3 Nil)))
-- = Cons (1 + 1) (Cons (2 + 1) (map (+1) (Cons 3 Nil)))
-- = Cons (1 + 1) (Cons (2 + 1) (Cons (3 + 1) (map (+1) Nil)))
-- = Cons (1 + 1) (Cons (2 + 1) (Cons (3 + 1) Nil))

head :: [] a -> a
head (a:as) = a
head [] = error "head"

tail :: [] a -> [] a
tail []     = error "tail"
tail (a:as) = as

ones :: [Int]
ones = 1:ones

-- filter even [1,2,3,4] == [2,4]
filter :: (a -> Bool) -> [a] -> [a]
filter p [] = []
filter p (x:xs)
  | p x       = x:resto
  | otherwise = resto
  where resto = filter p xs

zip []     ys     = []
zip (a:as) []     = []
zip (a:as) (b:bs) = (a, b):zip as bs

foldr :: (a -> r -> r) -> r
      -> [a] -> r
-- f :: a -> r -> r
-- z :: r
-- xs :: [a]
foldr f z []     = z
foldr f z (x:xs) = f x (foldr f z xs)
  -- x :: a
  -- xs :: [a] 

length_2 :: [a] -> Int
length_2 xs = foldr f z xs where
  f :: a -> Int -> Int
  f x length_tail = 1 + length_tail

  z :: Int
  z = 0

map_2 :: (a -> b) -> [a] -> [b]
map_2 a_em_b xs = foldr f z xs where
  f a lista_de_bs = a_em_b a:lista_de_bs

  z = []

filter_2 :: (a -> Bool) -> [a] -> [a]
filter_2 predicado xs = foldr f z xs where
  f a lista_filtrada
    | predicado a = a:lista_filtrada
    | otherwise = lista_filtrada
  
  z = []

-- filter_2 even [1,2,3,4]
-- = foldr f z [1,2,3,4]
-- = f 1 (foldr f z [2,3,4])
-- = foldr f z [2,3,4]
-- f 2 (foldr f z [3,4])
-- 2:foldr f z [3,4]
-- 2:f 3 (foldr f z [4])
-- 2:foldr f z [4]
-- 2:4:foldr f z []
-- 2:4:[]
-- [2,4]

fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci 2 = 1
fibonacci c = fibonacci (c - 1) + fibonacci (c - 2)

zip     ::                  [a] -> [b] -> [(a, b)]
zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith f [] _ = []
zipWith f _ [] = []
zipWith f (x:xs) (y:ys) = f x y:zipWith f xs ys

-- [1, 2, 3, 4]
--  +  +  +  +
-- [2, 3, 4, 5]
-- [3, 5, 7, 9]

fibs :: [Integer]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)
-- fibs = 0 : 1 : 1 : ...
-- fibs = 0 : 1 : 1 : 2 : ...

-- head fibs = 0
-- tail fibs = 1 : zipWith (+) fibs (tail fibs)
--             ^ fibs'
-- head fibs' = 1
-- tail fibs' = zipWith (+) fibs (tail fibs)
--            ^ resto
-- head resto = head (zipWith (+) (0:1:resto) (1:resto))
--              head ((0+1):zipWith (+) (1:resto) resto)
--              1
-- tail resto = zipWith (+) 1:(1:resto') (1:resto')
--              (1+1):zipWith (+) (1:resto'') resto''
--              zipWith (+) (1:2:resto'') (2:resto'')  
--              (1+2): ....

-- data Maybe a = Just a | Nothing

maybe :: (a -> r) -> r -> Maybe a -> r
maybe f defaulte Nothing = defaulte
maybe f _ (Just x)       = f x

divisao :: Int -> Int -> Maybe Int
divisao _ 0 = Nothing
divisao x y = Just (x `div` y)

bind :: Maybe a -> (a -> Maybe b) -> Maybe b
bind Nothing f  = Nothing
bind (Just a) f = f a

valida :: Int -> Maybe Int
valida x
  | x `mod` 2 == 0 = Just x
  | otherwise = Nothing

f x y = x `f` y

-- bind           :: Maybe a -> (a -> Maybe b) -> Maybe b
-- concatMap      :: [] a    -> (a -> []    b) -> []    b
-- (>>=)          :: IO a    -> (a -> IO    b) -> IO    b

concatena :: [a] -> [a] -> [a]
concatena [] ys = ys
concatena (x:xs) ys = x:(concatena xs ys)

concatMap :: [a] -> (a -> [b]) -> [b]
concatMap [] f = []
concatMap (x:xs) f = f x `concatena` concatMap xs f

-- concatMap f xs = concat (map f xs)
--                  ^^ [[a]] -> [a]

flatten :: Maybe (Maybe a) -> Maybe a
flatten xs = _

maybeMap :: (a -> b) -> Maybe a -> Maybe b
maybeMap f xs = _

-- bind xs f = flatten (maybeMap f xs)