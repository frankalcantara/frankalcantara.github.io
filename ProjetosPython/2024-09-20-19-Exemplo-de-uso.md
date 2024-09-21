---
author: Frank
beforetoc: '[Anterior](2024-09-20-18-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-20-Sem-T%C3%ADtulo.md)'
categories:
- Matemática
- Linguagens Formais
- Programação
description: Dynamic Programming in C++ with practical examples, performance analysis,
  and detailed explanations to optimize your coding skills and algorithm efficiency.
draft: null
featured: true
image: assets/images/prog_dynamic.jpeg
keywords:
- Dynamic Programming
- C++ Algorithms
- Coding Examples
- Performance Optimization
- Algorithm Efficiency
- Programming Guide
- Code Comparison
- Developer Tips
lastmod: 2024-09-20 22:35:36.353000+00:00
layout: post
preview: In this comprehensive guide, we delve into the world of Dynamic Programming
  with C++. Learn the core principles of Dynamic Programming, explore various algorithmic
  examples, and understand performance differences through detailed code comparisons.
  Perfect for developers looking to optimize their coding skills and enhance algorithm
  efficiency.
published: 2024-06-27 19:43:15.124000+00:00
rating: 5
slug: competitive-programming-techniques-insights
tags:
- Matemática
- Linguagens Formais
- Programação Dinâmica
- Dynamic Programming
- C++ Algorithms
- Performance Analysis
- Coding Examples
- Algorithm Optimization
- Practical Programming Guide
title: 'Exemplo de uso:'
toc: true
---
# Exemplo de uso:
humidity = [45, 52, 33, 64]
adjustments = [[5, 0], [-20, 1], [-14, 0], [18, 3]]
result = calculate_even_sum_after_adjustments(humidity, adjustments)
print(result)  # Saída: [166, 146, 132, 150]
```

**Implementation**: C++ 20

```cpp
#include <iostream>  // Includes the library for input and output operations.
#include <vector>    // Includes the library to use vectors.
#include <numeric>   // Includes the library that provides the accumulate function.

using namespace std;

// Function that adjusts the humidity levels and calculates the sum of even values after each adjustment.
vector<long long> adjustHumidity(vector<int>& humidity, const vector<vector<int>>& adjustments) {
    // Creates a vector to store the results, reserving enough space to avoid unnecessary reallocations.
    vector<long long> result;
    result.reserve(adjustments.size());

    // Iterates over each adjustment provided.
    for (const auto& adjustment : adjustments) {
        int value = adjustment[0];  // Extracts the adjustment value.
        int index = adjustment[1];  // Extracts the sensor index to be adjusted.

        // Updates the value in humidity[index] with the adjustment.
        humidity[index] += value;

        // Calculates the sum of even values in the humidity array after the update.
        long long sum = accumulate(humidity.begin(), humidity.end(), 0LL,
            [](long long acc, int val) {
                return acc + (val % 2 == 0 ? val : 0);  // Adds to the sum if the value is even.
            });

        // Adds the current sum of even values to the result vector.
        result.push_back(sum);
    }
    // Returns the vector containing the sum of even values after each adjustment.
    return result;
}

// Helper function to print the results in a formatted way.
void printResult(const vector<int>& humidity, const vector<vector<int>>& adjustments, const vector<long long>& result) {
    // Prints the initial humidity and the adjustments.
    cout << "**Input**: humidity = [";
    for (size_t i = 0; i < humidity.size(); ++i) {
        cout << humidity[i] << (i < humidity.size() - 1 ? ", " : "");  // Prints each humidity value, separating them with commas.
    }
    cout << "], adjustments = [";
    for (size_t i = 0; i < adjustments.size(); ++i) {
        // Prints each adjustment in the form [value, index].
        cout << "[" << adjustments[i][0] << "," << adjustments[i][1] << "]" << (i < adjustments.size() - 1 ? ", " : "");
    }
    cout << "]\n";

    // Prints the result after each adjustment.
    cout << "**Output**: ";
    for (long long res : result) {
        cout << res << " ";  // Prints each result, separating them by spaces.
    }
    cout << "\n\n";
}

int main() {
    // Example 1
    vector<int> humidity1 = { 45, 52, 33, 64 };  // Initial humidity vector.
    vector<vector<int>> adjustments1 = { {5,0}, {-20,1}, {-14,0}, {18,3} };  // Adjustment vector.
    cout << "Example 1:\n";
    auto result1 = adjustHumidity(humidity1, adjustments1);  // Calculates the results.
    printResult(humidity1, adjustments1, result1);  // Prints the results.

    // Example 2
    vector<int> humidity2 = { 40 };  // Initial humidity vector.
    vector<vector<int>> adjustments2 = { {12,0} };  // Adjustment vector.
    cout << "Example 2:\n";
    auto result2 = adjustHumidity(humidity2, adjustments2);  // Calculates the results.
    printResult(humidity2, adjustments2, result2);  // Prints the results.

    return 0;  // Indicates that the program terminated successfully.
}
```

The only noteworthy fragment in previous C++ implementation is the lambda function used to calculate the sum in:

```cpp
// Calculates the sum of even values in the humidity array after the update.
        long long sum = accumulate(humidity.begin(), humidity.end(), 0LL,
            [](long long acc, int val) {
                return acc + (val % 2 == 0 ? val : 0);  // Adds to the sum if the value is even.
            });
```

This line calculates the sum of even values in the `humidity` array after the update. The `accumulate` function is used to iterate over the `humidity` array and sum only the even values. The first two parameters, `humidity.begin()` and `humidity.end()`, define the range of elements in the array to be processed. The third parameter, `0LL`, initializes the accumulator with a value of $0$, where `LL` specifies that it is a `long long` integer.

The fourth parameter is a lambda function that takes two arguments: `acc`, which is the accumulated sum so far, and `val`, the current value being processed from the array. Inside the lambda function, the expression `val % 2 == 0 ? val : 0` checks whether the current value `val` is even (i.e., divisible by 2). If `val` is even, it is added to the accumulator `acc`; otherwise, 0 is added, which does not affect the sum.

Thus, the final result of the `accumulate` function is the sum of only the even values in the array, which is then stored in the variable `sum`. Well, something needs a little bit of attention.

> The `<numeric>` library in C++ provides a collection of functions primarily focused on numerical operations. These functions are designed to simplify common tasks such as accumulating sums, performing inner products, calculating partial sums, and more. One of the most commonly used functions in this library is `accumulate`, which is used to compute the sum (or other types of accumulation) of a range of elements in a container.
>
> The general syntax for the `accumulate` function is:
>
> ```cpp
> T accumulate(InputIterator first, InputIterator last, T init);
> T accumulate(InputIterator first, InputIterator last, T init, BinaryOperation op);
> ```
>
> - **InputIterator first, last**: These define the range of elements to be accumulated. The `first` points to the beginning of the range, and `last` points to one past the end of the range.
> - **T init**: This is the initial value of the accumulator, where the result of the accumulation will start.
> - **BinaryOperation op** _(optional)_: This is an optional custom function (usually a lambda or function object) that specifies how two elements are combined during the accumulation. If not provided, the function defaults to using the addition operator (`+`).
>
> **Example 1**: Simple Accumulation (Summing Elements): In its simplest form, `accumulate` can be used to sum all elements in a range.
>
> ```cpp
> #include <numeric>
> #include <vector>
> #include <iostream>
>
> int main() {
>     std::vector<int> vec = {1, 2, 3, 4, 5};
>     int sum = std::accumulate(vec.begin(), vec.end(), 0);  // Sum of all elements
>     std::cout << sum;  // Outputs: 15
>     return 0;
> }
> ```
>
> In this example, `accumulate` is used with the addition operator (default behavior) to sum the elements in the vector.
>
> **Example 2**: Custom Accumulation Using a Lambda Function: A custom operation can be applied during accumulation by providing a binary operation. For instance, to multiply all elements instead of summing them:
>
> ```cpp
> int product = std::accumulate(vec.begin(), vec.end(), 1, [](int acc, int x) {
>     return acc * x;
> });
> std::cout << product;  // Outputs: 120
> ```
>
> Here, instead of summing, the lambda function multiplies the elements.
>
> **Key Features of `accumulate`**:
>
> - **Default behavior**: When no custom operation is provided, `accumulate` simply adds the elements of the range, starting with the initial value.
> - **Custom operations**: By passing a custom binary operation, `accumulate` can perform more complex operations like multiplication, finding the maximum, or applying transformations.
> - **Initial value**: The initial value is critical for defining the result type and the starting point of the accumulation. For instance, starting the accumulation with `0` results in a sum, while starting with `1` can be useful for calculating products.
>
> **Example 3**: Accumulating with Different Types
> `accumulate` can also work with different types by adjusting the initial value and operation. For example, accumulating floating-point values from integers:
>
> ```cpp
> double avg = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
> std::cout << avg;  // Outputs: 3.0
> ```
>
> In this case, starting the accumulation with a double (`0.0`) ensures that the result is a floating-point number.
>
> **Limitations and Considerations**:
>
> - **No built-in parallelism**: The standard `accumulate` function does not support parallel execution, meaning it processes elements sequentially. For parallel processing, alternative solutions like algorithms from the `<execution>` library introduced in C++17 are required.
> - **Performance**: The time complexity of `accumulate` is $O(n)$, as it iterates over each element exactly once, applying the operation specified.
>
> **Example 4**: Custom Accumulation to Filter Elements
> You can use `accumulate` in combination with a lambda to perform conditional accumulation. For example, to sum only even numbers:
>
> ```cpp
> int even_sum = std::accumulate(vec.begin(), vec.end(), 0, [](int acc, int x) {
>     return (x % 2 == 0) ? acc + x : acc;
> });
> std::cout << even_sum;  // Outputs: 6 (2 + 4)
> ```
>
> In this example, only the even numbers are added to the sum by applying a condition within the lambda function.

Finally, we need to clarify lambda functions in C++ 20.

> **Lambda functions** in C++, available since C++ 11, are anonymous functions, meaning they do not have a name like regular functions. These are used when a function is needed only temporarily, typically for short operations, such as inline calculations or callback functions. Lambda functions are defined in place where they are used and can capture variables from their surrounding scope. Lambdas in C++ have been available since C++11, but in C++20, their capabilities were further expanded, making them more powerful and flexible.
>
> The general syntax for a lambda function in C++ is as follows:
>
> ```cpp
> [capture](parameters) -> return_type {
>     // function body
> };
> ```
>
> - **Capture**: Specifies which variables from the surrounding scope can be used inside the lambda. Variables can be captured by value `[=]` or by reference `[&]`. You can also specify individual variables, such as `[x]` or `[&x]`, to capture them by value or reference, respectively.
> - **Parameters**: The input parameters for the lambda function, similar to function arguments.
> - **Return Type**: Optional in most cases, as C++ can infer the return type automatically. However, if the return type is ambiguous or complex, it can be specified explicitly using `-> return_type`.
> - **Body**: The actual code to be executed when the lambda is called.
>
> C++20 brought some new features to lambda functions. One of the most important improvements is the ability to use lambdas in **immediate functions** (with `consteval`), and lambdas can now be default-constructed without capturing any variables. Additionally, lambdas in C++20 can use **template parameters**, allowing them to be more flexible and generic.
>
> **Example 1**: Basic Lambda Function: A simple example of a lambda function that sums two numbers:
>
> ```cpp
> auto sum = [](int a, int b) -> int {
>     return a + b;
> };
> std::cout << sum(5, 3);  // Outputs: 8
> ```
>
> **Example 2**: Lambda with Capture: In this example, a variable from the surrounding scope is captured by value:
>
> ```cpp
> int x = 10;
> auto multiply = [x](int a) {
>     return x * a;
> };
> std::cout << multiply(5);  // Outputs: 50
> ```
>
> Here, the lambda captures `x` by value and uses it in its body.
>
> **Example 3**: Lambda with Capture by Reference: In this case, the variable `y` is captured by reference, allowing the lambda to modify it:
>
> ```cpp
> int y = 20;
> auto increment = [&y]() {
>     y++;
> };
> increment();
> std::cout << y;  // Outputs: 21
> ```
>
> **Example 4**: Generic Lambda Function with C++20: With C++20, lambdas can now use template parameters, making them more generic:
>
> ```cpp
> auto generic_lambda = []<typename T>(T a, T b) {
>     return a + b;
> };
> std::cout << generic_lambda(5, 3);      // Outputs: 8
> std::cout << generic_lambda(2.5, 1.5);  // Outputs: 4.0
> ```
>
> This lambda can add both integers and floating-point numbers by utilizing template parameters.
>
> **Key Improvements in C++20**:
>
> - **Default-constructed lambdas**: In C++20, lambdas that do not capture any variables can now be _default-constructed_. This means they can be created and assigned to a variable without being immediately invoked or fully defined. This allows storing and passing lambdas for later use when default behavior is required.
>
>   ```cpp
>   auto default_lambda = [] {};  // Define a lambda with no capture or parameters
>   default_lambda();             // Call the lambda; valid as of C++20
>   ```
>
>   This feature enables the initialization of lambdas for deferred execution.
>
> - **Immediate lambdas**: C++20 introduces **consteval**, which ensures that functions marked with this keyword are evaluated at compile-time. When used with lambdas, this feature guarantees that the lambda's execution happens during compilation, and the result is already known by the time the program runs. A lambda used within a `consteval` function enforces compile-time evaluation.
>
>   **In programming competitions, `consteval` lambdas are unlikely to be useful because contests focus on runtime performance, and compile-time evaluation does not offer any competitive advantage. Problems in contests rarely benefit from compile-time execution, as the goal is typically to optimize runtime efficiency.**
>
>   **Consteval** ensures that the function cannot be executed at runtime. If a function marked `consteval` is invoked in a context that does not allow compile-time evaluation, it results in a compile-time error.
>
>   Example:
>
>   ```cpp
>   consteval auto square(int x) {
>       return [] (int y) { return y * y; }(x);
>   }
>   int value = square(5);  // Computed at compile-time
>   ```
>
>   In this example, the lambda inside the `square` function is evaluated at compile-time, producing the result before the program starts execution.
>
>   **Since programming contests focus on runtime behavior and dynamic inputs, features like `consteval` are not typically useful. Compile-time operations are not usually required in contests, where inputs are provided after the program has already started executing.**
>
> - **Template lambdas**: C++20 allows lambdas to accept **template parameters**, enabling generic behavior. This feature lets lambdas handle different data types without the need for function overloads or separate template functions. The template parameter is declared directly in the lambda's definition, allowing the same lambda to adapt to various types.
>
>   Example:
>
>   ```cpp
>   auto generic_lambda = []<typename T>(T a, T b) {
>       return a + b;
>   };
>   std::cout << generic_lambda(5, 3);      // Outputs: 8
>   std::cout << generic_lambda(2.5, 1.5);  // Outputs: 4.0
>   ```
>
>   In this case, the lambda can process both integer and floating-point numbers, dynamically adapting to the types of its arguments.

###### Data Type Analysis in the `adjustHumidity` Function

The choice of `long long` for the return type of the `adjustHumidity` function and for storing intermediate sums is made to ensure safety and prevent overflow in extreme cases:

- **Array size**: The problem specifies that there can be up to $10^4$ elements in the humidity array.
- **Maximum element value**: Each element in the array can have a value of up to $10^4$.
- **Worst-case scenario**: If all elements in the array are even and have the maximum value, the sum would be $10^4 \times 10^4 = 10^8$.
- **`int` limit**: In most implementations, an `int` has 32 bits, with a maximum value of $2^{31} - 1 ≈ 2.15 \times 10^9$.
- **Safety margin**: Although $10^8$ fits within an `int`, it is best practice to leave a safety margin, especially considering there may be multiple adjustments that could further increase the values.
- **`long long` guarantee**: A `long long` is guaranteed to be at least 64 bits, providing a much larger range (up to $2^{63} - 1$ for `signed long long`), which is more than sufficient for this problem.

By using `long long`, we ensure that no overflow occurs, even in extreme or unexpected cases. However, this could potentially lead to higher memory usage, which may exceed the limits in some competitive programming environments, depending on memory constraints.

###### Time Complexity Analysis

The current implementation recalculates the sum of even numbers in the `humidity` array after each adjustment using the `std::accumulate` function. This results in a time complexity of $O(n \times m)$, where $n$ is the size of the `humidity` array and $m$ is the number of adjustments in the `adjustments` list.

- **Accumulation per adjustment**: For each adjustment, the `std::accumulate` function iterates over all `n` elements in the `humidity` array. This operation takes $O(n)$ time.
- **Total complexity**: Since there are $m$ adjustments, the overall time complexity becomes $O(n \times m)$. This approach is inefficient for large values of $n$ and $m$ (e.g., if both $n$ and $m$ approach $10^4$), leading to performance issues in cases where the number of elements or adjustments is large.

###### Space Complexity Analysis

The space complexity is primarily influenced by the size of the input arrays:

- **Humidity array**: The `humidity` array contains $n$ elements, each of which is an `int`, so the space required for this array is $O(n)$.
- **Adjustments array**: The `adjustments` array contains $m$ adjustments, where each adjustment is a pair of integers. Therefore, the space required for this array is $O(m)$.
- **Result array**: The `result` vector stores $m$ results, each of type `long long`, so the space required for this vector is $O(m)$.

In total, the space complexity is $O(n + m)$.

The usage of `long long` ensures that the results and intermediate sums are safe from overflow, but it may slightly increase memory usage compared to using `int`. The overall space requirements are manageable within typical constraints in competitive programming environments, where both $n$ and $m$ are capped at $10^4$.

##### Algorithm for a Slightly Less Naive Code

1. Initialization:

   - Create a variable `even_sum` initialized to 0.
   - Create an empty list `results` to store the sums of even values after each adjustment.

2. Initial calculation of the sum of even values:

   - For each value $h$ in the `humidity` array:
     - If $h$ is even (i.e., $h \bmod 2 = 0$), add $h$ to `even_sum`.

3. For each adjustment $[adjustment_value, sensor\_index]$ in the `adjustments` list:

   a. Check if the current value in `humidity[sensor\_index]` is even:

   - If it is, subtract it from `even_sum`.

   b. Update the sensor's value:
   $humidity[sensor\_index] = humidity[sensor\_index] + adjustment_value$

   c. Check if the new value in `humidity[sensor\_index]` is even:

   - If it is, add it to `even_sum`.

   d. Add the current value of `even_sum` to the `results` list.

4. Return the `results` list.

**Code 2**:

```cpp
#include <iostream>
#include <vector>

using namespace std;

// Function that adjusts humidity levels and calculates the sum of even values after each adjustment.
vector<long long> adjustHumidity(vector<int>& humidity, const vector<vector<int>>& adjustments) {
    // Initialize the sum of even numbers to zero.
    long long sum = 0;

    // Calculate the initial sum of even values in the humidity array.
    for (int h : humidity) {
        if (h % 2 == 0) {  // Check if the value is even.
            sum += h;  // Add to the sum if it's even.
        }
    }

    // Create a vector to store the results, reserving enough space to avoid unnecessary reallocations.
    vector<long long> result;
    result.reserve(adjustments.size());

    // Iterate through each adjustment provided.
    for (const auto& adjustment : adjustments) {
        int value = adjustment[0];  // Extract the adjustment value.
        int index = adjustment[1];  // Extract the index of the sensor to be adjusted.

        // Check if the current value in humidity[index] is even.
        if (humidity[index] % 2 == 0) {
            sum -= humidity[index];  // If it's even, subtract it from the sum of even numbers.
        }

        // Update the value in humidity[index] with the adjustment.
        humidity[index] += value;

        // Check if the new value in humidity[index] is even after the update.
        if (humidity[index] % 2 == 0) {
            sum += humidity[index];  // If it's even, add it to the sum of even numbers.
        }

        // Add the current sum of even values to the result vector.
        result.push_back(sum);
    }

    // Return the vector containing the sum of even values after each adjustment.
    return result;
}

// Helper function to print the results in a formatted way.
void printResult(const vector<int>& humidity, const vector<vector<int>>& adjustments, const vector<long long>& result) {
    // Print the initial humidity values and the adjustments.
    cout << "**Input**: humidity = [";
    for (size_t i = 0; i < humidity.size(); ++i) {
        cout << humidity[i] << (i < humidity.size() - 1 ? ", " : "");
    }
    cout << "], adjustments = [";
    for (size_t i = 0; i < adjustments.size(); ++i) {
        cout << "[" << adjustments[i][0] << "," << adjustments[i][1] << "]" << (i < adjustments.size() - 1 ? ", " : "");
    }
    cout << "]\n";

    // Print the result after each adjustment.
    cout << "**Output**: ";
    for (long long res : result) {
        cout << res << " ";
    }
    cout << "\n\n";
}

int main() {
    // Example 1
    vector<int> humidity1 = { 45, 52, 33, 64 };  // Initial humidity array.
    vector<vector<int>> adjustments1 = { {5,0}, {-20,1}, {-14,0}, {18,3} };  // Adjustment array.
    cout << "Example 1:\n";
    auto result1 = adjustHumidity(humidity1, adjustments1);  // Compute the results.
    printResult(humidity1, adjustments1, result1);  // Print the results.

    // Example 2
    vector<int> humidity2 = { 40 };  // Initial humidity array.
    vector<vector<int>> adjustments2 = { {12,0} };  // Adjustment array.
    cout << "Example 2:\n";
    auto result2 = adjustHumidity(humidity2, adjustments2);  // Compute the results.
    printResult(humidity2, adjustments2, result2);  // Print the results.

    return 0;  // Indicate that the program completed successfully.
}
```

The Code 2 adjusts the humidity levels in an array and computes the sum of even numbers after each adjustment. It begins by initializing the sum of even numbers from the `humidity` array, adding each even element to a running total. This sum is stored in the variable `sum`, which is later updated based on adjustments made to the `humidity` array.

For each adjustment in the `adjustments` list, the code checks if the value at the target sensor index (i.e., `humidity[index]`) is even. If it is, that value is subtracted from the running total. After updating the sensor's value, the code checks again if the new value is even and adds it to the total if true. This ensures that only even numbers are considered in the running total, which is then stored in a results vector after each adjustment.

Finally, the results vector is returned, which contains the sum of even numbers in the `humidity` array after each adjustment. The `printResult` function is used to display the initial humidity values, the adjustments applied, and the resulting sums in a formatted manner.

> The `auto` keyword in C++ is used to automatically deduce the type of a variable at compile-time. This feature has been available since C++11, but with C++20, its functionality has been further enhanced, allowing for greater flexibility in template functions, lambdas, and other contexts where type inference can simplify code. The `auto` keyword is particularly useful when dealing with complex types, such as iterators, lambdas, or template instantiations, as it reduces the need for explicitly specifying types.
>
> When declaring a variable with `auto`, the type is inferred from the initializer. This eliminates the need to explicitly specify the type, which can be especially useful when working with types that are long or difficult to express.
>
> ```cpp
> auto x = 10;         // x is automatically deduced as an int
> auto y = 3.14;       // y is deduced as a double
> auto str = "Hello";  // str is deduced as a const char*
> ```
>
> In each case, the type of the variable is inferred based on the assigned value. This helps make code more concise and easier to maintain.
>
> **`auto` and Functions**:
>
> In C++20, the `auto` keyword can be used in function return types and parameters. The compiler deduces the return type or parameter type, allowing for greater flexibility in function definitions, especially with lambdas and template functions.
>
> **Example**:
>
> ```cpp
> auto add(auto a, auto b) {
>    return a + b;
> }
>
> int main() {
>    std::cout << add(5, 3);       // Outputs: 8
>    std::cout << add(2.5, 1.5);   // Outputs: 4.0
> }
> ```
>
> In this example, the `add` function can handle both integer and floating-point numbers because the types are deduced automatically. This simplifies function declarations, especially in template-like contexts.
>
> **`auto` with Lambdas and Template Functions**:
>
> C++20 allows for more complex use cases of `auto` within lambdas and template functions. For instance, lambda expressions can use `auto` to deduce parameter types without explicitly specifying them. Additionally, the `auto` keyword can be combined with template parameters to create generic, flexible code.
>
> **Example**:
>
> ```cpp
> auto lambda = [](auto a, auto b) {
>     return a + b;
> };
>
> std::cout << lambda(5, 3);        // Outputs: 8
> std::cout << lambda(2.5, 1.5);    // Outputs: 4.0
> ```
>
> Here, the lambda function uses `auto` to deduce the types of its parameters, making it applicable to both integers and floating-point numbers.

##### A Parallel Competitive Code

Using parallel code in this problem offers a advantage by allowing the calculation of the sum of even humidity values to be distributed across multiple processing threads. This can improve performance, especially for large humidity arrays, as the `reduce` function could leverage parallel execution policies to sum even values concurrently, reducing overall runtime. However, in the current implementation, the sequential execution policy (`exec_seq`) is used to maintain order. Additionally, the Code 3 already employs techniques to reduce verbosity, such as type aliases (`vi`, `vvi`, `vll`) and the use of `auto` for type deduction, making the code cleaner and easier to maintain without sacrificing readability.

In ICPC programming competitions, extremely large input arrays are not typically common, as problems are designed to be solvable within strict time limits, often with manageable input sizes. However, in other competitive programming environments, such as online coding platforms or specific algorithm challenges, larger datasets may appear, requiring more optimized solutions. These scenarios may involve parallel processing techniques or more efficient algorithms to handle the increased computational load. While this problem's input size is moderate, the techniques used here, like reducing verbosity with type aliases and utilizing `reduce`, ensure that the code can scale if needed.

Code 3 is already optimized to minimize function overhead, which can be an important factor in competitive programming. For instance, the entire algorithm is placed inside the `main` function, reducing the need for additional function calls and thus improving performance in time-sensitive environments.

**Code 3**:

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <execution>  // Necessary for execution policies in reduce

using namespace std;

// Aliases to reduce typing of long types
using vi = vector<int>;           // Alias for vector<int>
using vvi = vector<vector<int>>;  // Alias for vector of vectors of int
using vll = vector<long long>;    // Alias for vector<long long>
using exec_seq = execution::sequenced_policy; // Alias for execution::seq (sequential execution)

// Helper function to print the results in a formatted way.
void printResult(const vi& humidity, const vvi& adjustments, const vll& result) {
    // Prints the initial humidity array and the adjustments array.
    cout << "**Input**: humidity = [";
    for (size_t i = 0; i < humidity.size(); ++i) {
        // Print each humidity value, separating them with commas.
        cout << humidity[i] << (i < humidity.size() - 1 ? ", " : "");
    }
    cout << "], adjustments = [";
    for (size_t i = 0; i < adjustments.size(); ++i) {
        // Print each adjustment as [value, index], separating them with commas.
        cout << "[" << adjustments[i][0] << "," << adjustments[i][1] << "]" << (i < adjustments.size() - 1 ? ", " : "");
    }
    cout << "]\n";

    // Prints the results after each adjustment.
    cout << "**Output**: ";
    for (auto res : result) {  // Using `auto` to automatically deduce the type (long long)
        cout << res << " ";    // Print each result followed by a space.
    }
    cout << "\n\n";
}

int main() {
    // Example 1: Initialize the humidity vector and the adjustments to be made.
    vi humidity1 = { 45, 52, 33, 64 };  // Initial humidity levels for each sensor.
    vvi adjustments1 = { {5,0}, {-20,1}, {-14,0}, {18,3} };  // Adjustments in format {adjustment value, sensor index}.

    // Create a vector to store the results, reserving space to avoid reallocation during execution.
    vll result1;
    result1.reserve(adjustments1.size());

    // Process each adjustment for the humidity array.
    for (const auto& adjustment : adjustments1) {
        int value = adjustment[0];  // Get the adjustment value.
        int index = adjustment[1];  // Get the index of the sensor to be adjusted.

        // Apply the adjustment to the corresponding humidity value.
        humidity1[index] += value;

        // Calculate the sum of even values in the humidity array using the `reduce` function.
        auto sum = reduce(
            exec_seq{},              // Use sequential execution policy to maintain order.
            humidity1.begin(),       // Start iterator of the humidity vector.
            humidity1.end(),         // End iterator of the humidity vector.
            0LL,                     // Initial sum is 0 (as long long to avoid overflow).
            [](auto acc, auto val) { // Lambda function to accumulate even numbers.
                return acc + (val % 2 == 0 ? val : 0);  // Add to the sum only if the value is even.
            }
        );

        // Store the current sum of even values after the adjustment in the result vector.
        result1.push_back(sum);
    }

    // Print the results for the first example.
    cout << "Example 1:\n";
    printResult(humidity1, adjustments1, result1);

    // Example 2: Initialize the second humidity vector and the adjustments.
    vi humidity2 = { 40 };  // Initial humidity levels for the second example.
    vvi adjustments2 = { {12,0} };  // Adjustments for the second example.

    // Create a vector to store the results.
    vll result2;
    result2.reserve(adjustments2.size());

    // Process each adjustment for the second humidity array.
    for (const auto& adjustment : adjustments2) {
        int value = adjustment[0];  // Get the adjustment value.
        int index = adjustment[1];  // Get the index of the sensor to be adjusted.

        // Apply the adjustment to the corresponding humidity value.
        humidity2[index] += value;

        // Calculate the sum of even values in the humidity array using `reduce`.
        auto sum = reduce(
            exec_seq{},              // Use sequential execution policy to maintain order.
            humidity2.begin(),       // Start iterator of the humidity vector.
            humidity2.end(),         // End iterator of the humidity vector.
            0LL,                     // Initial sum is 0 (as long long to avoid overflow).
            [](auto acc, auto val) { // Lambda function to accumulate even numbers.
                return acc + (val % 2 == 0 ? val : 0);  // Add to the sum only if the value is even.
            }
        );

        // Store the current sum of even values after the adjustment in the result vector.
        result2.push_back(sum);
    }

    // Print the results for the second example.
    cout << "Example 2:\n";
    printResult(humidity2, adjustments2, result2);

    return 0;  // Indicate that the program finished successfully.
}
```

The core of the algorithm in Code 3 focuses on adjusting humidity levels based on a series of adjustments and then calculating the sum of even humidity values after each adjustment. The main part responsible for solving the problem involves iterating over each adjustment and performing two key operations: updating the humidity values and calculating the sum of even numbers in the updated array. This is done by:

1. **Adjusting the Humidity**: For each adjustment (which consists of an adjustment value and an index), the corresponding humidity value is updated by adding the adjustment value. This modifies the sensor reading at the specified index in the `humidity` vector.

   Example:

   ```cpp
   humidity[index] += value;
   ```

   This line updates the humidity value at the sensor located at `index` by adding the provided `value`.

2. **Calculating the Sum of Even Values**: After each adjustment, the algorithm calculates the sum of the even values in the `humidity` array. This is done using the `reduce` function with a lambda function that filters and sums only the even numbers. The key here is that the algorithm iterates over the entire `humidity` array and sums the values that are divisible by 2.

   Example:

   ```cpp
   auto sum = reduce(
       exec_seq{},              // Sequential execution
       humidity.begin(),        // Start of the humidity array
       humidity.end(),          // End of the humidity array
       0LL,                     // Initial sum set to 0 (long long)
       [](auto acc, auto val) { // Lambda to sum even values
           return acc + (val % 2 == 0 ? val : 0);
       }
   );
   ```

   This code calculates the sum of all even values in the `humidity` array after each adjustment, ensuring that only even numbers contribute to the total sum.

3. **Storing and Printing Results**: After calculating the sum of even values for each adjustment, the result is stored in a `result` vector, which is later printed to display the output. The `printResult` function is used to format and output the humidity values, adjustments, and the resulting sum of even values after each adjustment.

In this context, the parallel version of `reduce` is particularly useful when dealing with large datasets, where summing or reducing values sequentially can be time-consuming. The key advantage of using `reduce` with a parallel execution policy is its ability to distribute the workload across multiple cores, significantly reducing the overall execution time.

When `reduce` is used with the `execution::par` policy, it breaks the range of elements into smaller chunks and processes them in parallel. This means that instead of iterating through the array in a single thread (as done with `execution::seq`), the work is split among multiple threads, each of which processes a part of the array concurrently.

**Parallel Execution Example**:

In the following example, the `reduce` function is used to sum an array of humidity values, utilizing the `execution::par` policy:

```cpp
auto parallel_sum = std::reduce(std::execution::par, humidity.begin(), humidity.end(), 0LL,
                                [](auto acc, auto val) {
                                    return acc + (val % 2 == 0 ? val : 0);  // Sum only even values
                                });
```

**How the parallel execution works**:

1. **Data Splitting**: The `humidity` array is divided into smaller chunks, and each chunk is processed by a separate thread.
2. **Concurrent Processing**: Each thread sums the even values in its respective chunk. The `execution::par` policy ensures that this happens in parallel, taking advantage of multiple CPU cores.
3. **Final Reduction**: Once all threads complete their tasks, the partial results are combined into a final sum, which includes only the even values from the original array.

By distributing the workload across multiple threads, the program can achieve significant performance improvements when the `humidity` array is large. This approach is particularly useful in competitive programming contexts where optimizing time complexity for large inputs can be crucial to solving problems within strict time limits.

> The `reduce` function, introduced in C++17, is part of the `<numeric>` library and provides a way to aggregate values in a range by applying a binary operation, similar to `accumulate`. However, unlike `accumulate`, `reduce` can take advantage of parallel execution policies, making it more efficient for large data sets when concurrency is allowed. In C++20, `reduce` gained even more flexibility, making it a preferred choice for operations that benefit from parallelism.
>
> **Basic Syntax of `reduce`**:
>
> The general syntax for `reduce` is as follows:
>
> ```cpp
> T reduce(ExecutionPolicy policy, InputIterator first, InputIterator last, T init);
> T reduce(ExecutionPolicy policy, InputIterator first, InputIterator last, T init, BinaryOperation binary_op);
> ```
>
> - **ExecutionPolicy**: This specifies the execution policy, which can be `execution::seq` (sequential execution), `execution::par` (parallel execution), or `execution::par_unseq` (parallel and vectorized execution).
> - **InputIterator first, last**: These define the range of elements to be reduced.
> - **T init**: The initial value for the reduction (e.g., 0 for summing values).
> - **BinaryOperation binary_op** (optional): A custom operation to apply instead of the default addition.
>
> **Example 1**: Basic Reduce with Sequential Execution: This example demonstrates a basic sum reduction with sequential execution:
>
> ```cpp
> #include <iostream>
> #include <vector>
> #include <numeric>
> #include <execution>  // Required for execution policies
>
> int main() {
> std::vector<int> vec = {1, 2, 3, 4, 5};
> auto sum = std::reduce(std::execution::seq, vec.begin(), vec.end(), 0);
> std::cout << "Sum: " << sum; // Outputs: 15
> return 0;
> }
> ```
>
> Here, the `reduce` function uses the `execution::seq` policy to ensure that the >reduction happens in a sequential order, summing the values from `vec`.
>
> **Example 2**: Custom Binary Operation: You can also provide a custom binary operation using a lambda function. In this case, the reduction will multiply the elements instead of summing them:
>
> ```cpp
> auto product = std::reduce(std::execution::seq, vec.begin(), vec.end(), 1,
>                            [](int a, int b) { return a * b; });
> std::cout << "Product: " << product;  // Outputs: 120
> ```
>
> In this example, `reduce` applies the custom binary operation (multiplication) to aggregate the values in `vec`.
>
> **Parallelism in `reduce`**:
>
> The major advantage of `reduce` over `accumulate` is its ability to handle parallel execution. Using the `execution::par` policy allows `reduce` to split the workload across multiple threads, significantly improving performance on large datasets:
>
> ```cpp
> auto parallel_sum = std::reduce(std::execution::par, vec.begin(), vec.end(), 0);
> ```
>
> This enables `reduce` to sum the elements in `vec` concurrently, improving efficiency on large arrays, especially in multi-core environments.

##### Finally, the code using Fenwick tree

I chose to write this code using as much modern C++ as possible. This means you will face two challenges. The first is understanding the Fenwick tree algorithm, and the second is understanding the C++ syntax. To help make this easier, I will explain the code block by block, highlighting each C++ feature and why I chose to write it this way.

**Code 4**:

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <syncstream>

using namespace std;
namespace fs = filesystem;

namespace config {
    enum class InputMethod { Hardcoded, Stdin, File };

    // Altere esta linha para mudar o método de entrada
    inline constexpr InputMethod input_method = InputMethod::Hardcoded;
}

using vi = vector<int>;
using vvi = vector<vector<int>>;
using vll = vector<long long>;

class BIT {
    vi tree;
    int n;

public:
    Fenwick tree(int size) : tree(size + 1), n(size) {}

    void update(int i, int delta) {
        for (++i; i <= n; i += i & -i) tree[i] += delta;
    }

    long long query(int i) const {
        long long sum = 0;
        for (++i; i > 0; i -= i & -i) sum += tree[i];
        return sum;
    }
};

vll adjustHumidity(vi& humidity, const vvi& adjustments) {
    int n = humidity.size();
    BIT bit(n);
    vll result;
    result.reserve(adjustments.size());

    auto updateBit = [&](int i, int old_val, int new_val) {
        if (!(old_val & 1)) bit.update(i, -old_val);
        if (!(new_val & 1)) bit.update(i, new_val);
        };

    for (int i = 0; i < n; ++i) {
        if (!(humidity[i] & 1)) bit.update(i, humidity[i]);
    }

    for (const auto& adj : adjustments) {
        int i = adj[1], old_val = humidity[i], new_val = old_val + adj[0];
        updateBit(i, old_val, new_val);
        humidity[i] = new_val;
        result.push_back(bit.query(n - 1));
    }

    return result;
}

void printResult(osyncstream& out, const vi& humidity, const vvi& adjustments, const vll& result) {
    out << "**Input**: humidity = [" << humidity[0];
    for (int i = 1; i < humidity.size(); ++i) out << ", " << humidity[i];
    out << "], adjustments = [";
    for (const auto& adj : adjustments)
        out << "[" << adj[0] << "," << adj[1] << "]" << (&adj != &adjustments.back() ? ", " : "");
    out << "]\n**Output**: ";
    for (auto res : result) out << res << " ";
    out << "\n\n";
}

pair<vi, vvi> readInput(istream& in) {
    vi humidity;
    vvi adjustments;
    int n, m;
    in >> n;
    humidity.resize(n);
    for (int& h : humidity) in >> h;
    in >> m;
    adjustments.resize(m, vi(2));
    for (auto& adj : adjustments) in >> adj[0] >> adj[1];
    return { humidity, adjustments };
}

void processInput(istream& in, osyncstream& out) {
    int t;
    in >> t;
    for (int i = 1; i <= t; ++i) {
        out << "Example " << i << ":\n";
        auto [humidity, adjustments] = readInput(in);
        auto result = adjustHumidity(humidity, adjustments);
        printResult(out, humidity, adjustments, result);
    }
}

int main() {
    osyncstream syncout(cout);
{% raw %}
    if constexpr (config::input_method == config::InputMethod::Hardcoded) {
        vector<pair<vi, vvi>> tests = {{{45, 52, 33, 64}, {{5,0}, {-20,1}, {-14,0}, {18,3}}},{{40}, {{12,0}}},{{30, 41, 55, 68, 72}, {{10,0}, {-15,2}, {22,1}, {-8,4}, {5,3}}}};
{% endraw %}
        for (int i = 0; i < tests.size(); ++i) {
            syncout << "Example " << i + 1 << ":\n";
            auto& [humidity, adjustments] = tests[i];
            auto result = adjustHumidity(humidity, adjustments);
            printResult(syncout, humidity, adjustments, result);
        }
    }
    else if constexpr (config::input_method == config::InputMethod::Stdin) {
        processInput(cin, syncout);
    }
    else if constexpr (config::input_method == config::InputMethod::File) {
        fs::path inputPath = "input.txt";
        if (fs::exists(inputPath)) {
            ifstream inputFile(inputPath);
            processInput(inputFile, syncout);
        }
        else {
            syncout << "Input file not found: " << inputPath << endl;
        }
    }
    else {
        syncout << "Invalid input method defined" << endl;
    }

    return 0;
}
```

The first thing you should notice is that I chose to include all three possible input methods in the same code. Obviously, in a competition, you wouldn't do that. You would include only the method that interests you. Additionally, I opted to use modern C++20 capabilities instead of using the old preprocessor directives (`#defines`). However, before diving into the analysis of Code 4, let's look at an example of what the `main` function would look like if we were using preprocessor directives.

```cpp
#include ...

// Define input methods
#define INPUT_HARDCODED 1
#define INPUT_STDIN 2
#define INPUT_FILE 3

// Select input method here
#define INPUT_METHOD INPUT_STDIN

// lot of code goes here

int main() {
    // Creates a synchronized output stream (osyncstream) to ensure thread-safe output to cout.
    osyncstream syncout(cout);

    // Check if the input method is defined as INPUT_HARDCODED using preprocessor directives.
#if INPUT_METHOD == INPUT_HARDCODED
    // Define a vector of pairs where each pair contains:
    // 1. A vector of humidity levels.
    // 2. A 2D vector representing adjustments (value, index) to be applied to the humidity levels.
{% raw %}
    vector<pair<vi, vvi>> tests = {
        {{45, 52, 33, 64}, {{5,0}, {-20,1}, {-14,0}, {18,3}}},
        {{40}, {{12,0}}},
        {{30, 41, 55, 68, 72}, {{10,0}, {-15,2}, {22,1}, {-8,4}, {5,3}}}
    };
{% endraw %}
    // Iterate over each hardcoded test case.
    for (int i = 0; i < tests.size(); ++i) {
        // Print the example number using synchronized output to avoid race conditions in a multithreaded context.
        syncout << "Example " << i + 1 << ":\n";

        // Extract the humidity vector and adjustments vector using structured bindings (C++17 feature).
        auto& [humidity, adjustments] = tests[i];

        // Call the adjustHumidity function to apply the adjustments and get the results.
        auto result = adjustHumidity(humidity, adjustments);

        // Print the humidity, adjustments, and the results using the printResult function.
        printResult(syncout, humidity, adjustments, result);
    }

    // If the input method is INPUT_STDIN, process input from standard input.
#elif INPUT_METHOD == INPUT_STDIN
    // Call processInput to read input from standard input and produce output.
    processInput(cin, syncout);

    // If the input method is INPUT_FILE, read input from a file.
#elif INPUT_METHOD == INPUT_FILE
    // Define the file path where the input data is expected.
    fs::path inputPath = "input.txt";

    // Check if the file exists at the specified path.
    if (fs::exists(inputPath)) {
        // If the file exists, open it as an input file stream.
        ifstream inputFile(inputPath);

        // Call processInput to read data from the input file and produce output.
        processInput(inputFile, syncout);
    } else {
        // If the file does not exist, print an error message indicating that the input file was not found.
        syncout << "Input file not found: " << inputPath << endl;
    }

    // If none of the above input methods are defined, print an error message for an invalid input method.
#else
    syncout << "Invalid INPUT_METHOD defined" << endl;
#endif

    // Return 0 to indicate successful program termination.
    return 0;
}
```

The code fragment uses **preprocessor directives** to switch between different input methods for reading data, based on a pre-defined configuration. This is done using `#define` statements at the top of the code and `#if`, `#elif`, and `#else` directives in the `main` function.

**Input Method Definitions**:

```cpp
#define INPUT_HARDCODED 1
#define INPUT_STDIN 2
#define INPUT_FILE 3
```

These `#define` statements assign integer values to three possible input methods:

- `INPUT_HARDCODED`: The input data is hardcoded directly into the program.
- `INPUT_STDIN`: The input data is read from standard input (`stdin`), such as from the console.
- `INPUT_FILE`: The input data is read from a file, typically stored on disk.

**Input Method Selection**:

```cpp
#define INPUT_METHOD INPUT_STDIN
```

This line selects the input method by defining `INPUT_METHOD`. In this case, it is set to `INPUT_STDIN`, meaning that the program will expect to read input from the console. Changing this to `INPUT_HARDCODED` or `INPUT_FILE` would switch the input source.

**Conditional Compilation (`#if`, `#elif`, `#else`)**:

The conditional compilation directives (`#if`, `#elif`, `#else`) are used to include or exclude specific blocks of code based on the value of `INPUT_METHOD`.

```cpp
#if INPUT_METHOD == INPUT_HARDCODED
    // Code for hardcoded input goes here
#elif INPUT_METHOD == INPUT_STDIN
    // Code for reading from standard input goes here
#elif INPUT_METHOD == INPUT_FILE
    // Code for reading from a file goes here
#else
    // Code for handling invalid input method goes here
#endif
```

- **`#if INPUT_METHOD == INPUT_HARDCODED`**: If the input method is hardcoded, a predefined set of test cases (humidity levels and adjustments) will be used.
- **`#elif INPUT_METHOD == INPUT_STDIN`**: If the input method is set to standard input, the program will read from the console.
- **`#elif INPUT_METHOD == INPUT_FILE`**: If the input method is set to file input, the program will attempt to read from a file (`input.txt`).
- **`#else`**: If an invalid `INPUT_METHOD` is defined, an error message is printed.

These preprocessor directives enable the program to easily switch between input methods without having to manually modify the logic inside `main`, providing flexibility depending on how the input is expected during execution. But, since we are using C++20, this might not be the best solution. It may be the fastest for competitions, but there is a fundamental reason why I'm making things a bit more complex here. Beyond just learning how to write code for competitions, we are also learning C++20. Let's start by:

The code starts by importing the `std` namespace globally with **`using namespace std;`, which allows using standard C++ objects (like `cout`, `vector`, etc.) without having to prefix them with `std::`**.
s

```cpp
using namespace std;  // Use the standard namespace to avoid typing "std::" before standard types.
```

The line **`namespace fs = filesystem;`** creates an alias for the `filesystem` namespace, allowing the code to reference `filesystem` functions more concisely, using `fs::` instead of `std::filesystem::`.

```cpp
namespace fs = filesystem;  // Alias for the filesystem namespace.
```

Inside the **`config` namespace**, there is an **enum class** `InputMethod` that defines three possible input methods: `Hardcoded`, `Stdin`, and `File`. This helps manage how input will be provided to the program.

```cpp
namespace config {
    enum class InputMethod { Hardcoded, Stdin, File };  // Enum to define input methods
```

> The **`namespace config`** is used to encapsulate related constants and configuration settings into a specific scope. In this case, it organizes the input methods and settings used in the program. By placing these within a namespace, we avoid cluttering the global namespace, ensuring that these settings are logically grouped together. This encapsulation makes it easier to maintain the code, preventing potential naming conflicts and allowing future expansion of the configuration without affecting other parts of the program.
>
> The **`namespace config`** does not come from the standard C++ library; it is created specifically within this code to group configurations like the `InputMethod`. The use of namespaces in C++ allows developers to organize code and avoid naming conflicts but is independent of the C++ Standard Library or language itself.
>
> The **`enum class InputMethod`** provides a strongly typed, scoped enumeration. Unlike traditional enums, an `enum class` does not implicitly convert its values to integers, which helps prevent accidental misuse of values. The scoped nature of `enum class` also means that its values are contained within the enumeration itself, avoiding naming conflicts with other parts of the program. For instance, instead of directly using `Hardcoded`, you use `InputMethod::Hardcoded`, making the code more readable and avoiding ambiguity.
>
> Here's an example of using an **enum class** in a small program. This example demonstrates how to select an input method based on the defined `InputMethod`:
>
> ```cpp
> #include <iostream>
>
> enum class InputMethod { Hardcoded, Stdin, File };
>
> void selectInputMethod(InputMethod method) {
>     switch (method) {
>         case InputMethod::Hardcoded:
>             std::cout << "Using hardcoded input.\n";
>             break;
>         case InputMethod::Stdin:
>             std::cout << "Reading input from stdin.\n";
>             break;
>         case InputMethod::File:
>             std::cout << "Reading input from a file.\n";
>             break;
>     }
> }
>
> int main() {
>     InputMethod method = InputMethod::File;
>     selectInputMethod(method);  // **Output**: Reading input from a file.
>     return 0;
> }
> ```
>
> In this example, the `enum class InputMethod` allows for a clear, type-safe way to represent the input method, making the code easier to manage and less error-prone.

The **`inline constexpr`** constant `input_method` specifies which input method will be used by default. In this case, it is set to `InputMethod::Hardcoded`, meaning the input will be predefined inside the code. The `inline constexpr` allows the value to be defined at compile time, making it a more efficient configuration option.

```cpp
    inline constexpr InputMethod input_method = InputMethod::Hardcoded;  // Default input method is hardcoded.
}
```

> The **`inline`** keyword in C++ specifies that a function, variable, or constant is defined inline, meaning the compiler should attempt to replace function calls with the actual code of the function. This can improve performance by avoiding the overhead of a function call. However, the main use of `inline` in modern C++ is to avoid the "multiple definition" problem when defining variables or functions in header files that are included in multiple translation units.
>
> ```cpp
> inline int square(int x) {
>     return x * x;  // This function is defined inline, so calls to square(3) may be replaced with 3 * 3 directly.
> }
> ```
>
> When `inline` is used with **variables or constants**, it allows those variables or constants to be defined in a header file without violating the One Definition Rule (ODR). Each translation unit that includes the header will have its own copy of the inline variable, but the linker will ensure that only one copy is used in the final binary.
>
> ```cpp
> inline constexpr int max_value = 100;  // This constant can be included in multiple translation units without causing redefinition errors.
> ```
>
> The **`constexpr`** keyword specifies that a function or variable can be evaluated at compile-time. It guarantees that, if possible, the function will be computed by the compiler, not at runtime. This is especially useful in optimization, as it allows constants to be determined and used during the compilation process rather than execution.
>
> **`constexpr` with Variables**:
> When you use `constexpr` with variables, the compiler knows that the variable's value is constant and should be computed at compile time.
>
> ```cpp
> constexpr int max_items = 42;  // The value of max_items is known at compile-time and cannot change.
> ```
>
> You can use `constexpr` variables to define array sizes or template parameters because their values are known during compilation.
>
> ```cpp
> constexpr int size = 10;
> int array[size];  // Valid, because size is a constant expression.
> ```
>
> **`constexpr` with Functions**:
> A **`constexpr` function** is a function whose return value can be computed at compile time if the inputs are constant expressions. The function must have a single return statement and all operations within it must be valid at compile time.
>
> ```cpp
> constexpr int factorial(int n) {
>     return n <= 1 ? 1 : n * factorial(n - 1);  // Recursive function that computes the factorial at compile time.
> }
> ```
>
> If `factorial(5)` is called with a constant value, the compiler will compute the result at compile time and replace the function call with `120` in the generated binary.
> ?
> **Combining `inline` and `constexpr`**:
> A function can be both **`inline`** and **`constexpr`**, which means the function can be evaluated at compile time and its calls may be inlined if appropriate.
>
> ```cpp
> inline constexpr int power(int base, int exp) {
>     return (exp == 0) ? 1 : base * power(base, exp - 1);
> }
> ```
>
> In this case, the `power` function will be inlined when called at runtime and computed at compile time if the arguments are constant. For example, `power(2, 3)` would be replaced by `8` at compile time.
>
> **Practical Use of `constexpr`**:
> `constexpr` can be used in a wide variety of contexts, such as constructing constant data, optimizing algorithms, and defining efficient compile-time logic. Here are a few examples:
>
> 1. **Compile-time array size**:
>
> ```cpp
>  constexpr int size = 5;
>  int array[size];  // The size is computed at compile time.
> ```
>
> 2. **Compile-time strings**:
>
> ```cpp
> constexpr const char* greet() { return "Hello, World!"; }
> constexpr const char* message = greet();  // The message is computed at compile time.
> ```
>
> 3. **Compile-time mathematical operations**:
>
> ```cpp
> constexpr int area(int length, int width) {
>     return length * width;
> }
> constexpr int room_area = area(10, 12);  // Computed at compile time.
> ```
>
> **Using `constexpr` in Competitive Programming**:
> In competitive programming, **`constexpr`** can be both an advantage and a disadvantage, depending on how it is used.
>
> - **Advantage**: `constexpr` can optimize code by computing results at compile time rather than runtime, which can save valuable processing time. For example, if you know certain values or calculations are constant throughout the competition, you can use `constexpr` to precompute them, thereby avoiding recalculations during execution.
>
> - **Disadvantage**: However, in many competitive programming problems, the input is dynamic and provided at runtime, meaning that `constexpr` cannot be used for computations that depend on this input. Since the focus in competitive programming is on runtime efficiency, the use of `constexpr` is limited to cases where you can precompute values before the competition or during compilation.
>
> Overall, `constexpr` is valuable when solving problems with static data or fixed input sizes, but in typical ICPC-style competitions, its usage may be less frequent because most problems require dynamic input processing.
>
> In summary, **`inline`** helps with reducing overhead by allowing the compiler to replace function calls with the actual function code, and it prevents multiple definitions of variables in multiple translation units. **`constexpr`** enables computations to be performed at compile time, which can significantly optimize performance by avoiding runtime calculations, although its applicability in competitive programming may be limited.

AINDA TEM MUITO QUE EXPLICAR AQUI.

### Algorithm: Incremental Sum

The **Incremental Sum Algorithm** offers an efficient method for maintaining a running sum of specific elements (such as even numbers) in an array while applying adjustments. This approach eliminates the need to recalculate the entire sum after each modification, instead updating the sum incrementally by subtracting old values and adding new ones as necessary.

The algorithm begins with an initial calculation of the sum of even numbers in the array. This step has a time complexity of $O(n)$, where $n$ represents the array size. For example, in Python, this initial calculation could be implemented as:

```python
def initial_sum(arr):
    return sum(x for x in arr if x % 2 == 0)
```

Following the initial calculation, the algorithm processes each adjustment to the array. For each adjustment, it performs three key operations: If the old value at the adjusted index was even, it subtracts this value from the sum. It then updates the array element with the new value. Finally, if the new value is even, it adds this value to the sum. This process maintains the sum's accuracy with a constant time complexity of $O(1)$ per adjustment. In C++, this adjustment process could be implemented as follows:

```cpp
void adjust(vector<int>& arr, int index, int new_value, int& even_sum) {
    if (arr[index] % 2 == 0) even_sum -= arr[index];
    arr[index] = new_value;
    if (new_value % 2 == 0) even_sum += new_value;
}
```

The algorithm's efficiency stems from its ability to process adjustments in constant time, regardless of the array's size. This approach is particularly beneficial when dealing with numerous adjustments, as it eliminates the need for repeated full array traversals.

To illustrate the algorithm's operation, consider the following example:

```python
arr = [1, 2, 3, 4, 5]
even_sum = initial_sum(arr)  # even_sum = 6 (2 + 4)

