{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -O0 #-}
module Main where

import Debug.Trace

insert :: Monad m => (a -> a -> m Bool) -> a -> [a] -> m [a]
insert _ x [] = pure [x]
insert cmp x (y:xs) = do
  menor <- cmp x y
  if menor
    then pure (x:y:xs)
    else do
      rest <- insert cmp x xs
      pure (y:rest)

insertion :: forall a m. Monad m
          => (a -> a -> m Bool)
          -> [a] -> m [a] 
insertion cmp [] = pure []
insertion cmp (x:xs) = do
  rest <- insertion cmp xs
  insert cmp x rest

merge :: forall a m. Monad m
      => (a -> a -> m Bool)
      -> [a] -> [a] -> m [a]
merge cmp [] x = pure x
merge cmp x [] = pure x
merge cmp (x:xs) (y:ys) = do
  menor <- cmp x y
  if menor
    then do
      resto <- merge cmp xs (y:ys)
      pure (x:resto)
    else do
      resto <- merge cmp (x:xs) ys
      pure (y:resto)

divide :: [a] -> ([a], [a])
divide = go True where
  go _ [] = ([], [])
  go True (x:xs) = let (a, b) = go False xs in (x:a, b)
  go False (x:xs) = let (a, b) = go True xs in (a, x:b)

mergeSort :: Monad m => (a -> a -> m Bool) -> [a] -> m [a]
mergeSort _ [] = pure []
mergeSort _ [x] = pure [x]
mergeSort cmp xs = do
  let (a, b) = divide xs
  a <- mergeSort cmp a 
  b <- mergeSort cmp b 
  merge cmp a b

coin :: a -> a -> [Bool]
coin _ _ = [True, False]

dobra x = trace "avaliando dobra" $ x + x

y = trace "avaliando y" $ dobra 2

main :: IO ()
main = do
  linha <- getLine
  let
    x :: Int 
    x = read linha
  print (read linha)

-- read :: Read a => String -> a
-- print :: Show a => a -> IO ()
