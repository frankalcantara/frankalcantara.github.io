---
author: Frank
beforetoc: |-
  [Anterior](2024-09-24-2-1.-Introduction.md)
  [Próximo](2024-09-24-4-4.-Introduction-to-File-IO-in-C%2B%2B.md)
categories:
  - Matemática
  - Linguagens Formais
  - Programação
description: Competitive Programming in C++ and Python for newbies.
draft: null
featured: false
image: assets/images/prog_dynamic.jpeg
keywords:
  - Code Comparison
  - Developer Tips
lastmod: 2024-09-28T02:52:37.111Z
layout: post
preview: Competitive Programming in C++ and Python for newbies. Explore various algorithmic examples, and understand performance differences. Perfect for developers looking to optimize their coding skills and boost algorithm efficiency.
published: false
rating: 5
slug: competitive-programming-techniques-insights
tags:
  - Matemática
  - Linguagens Formais
title: 2. C++ Competitive Programming Hacks
toc: true
---

# 2. C++ Competitive Programming Hacks

This chapter gives you practical steps to improve your speed and performance in competitive programming with C++ 20. C++ is fast and powerful, but using it well takes skill and focus. We will cover how to type faster, write cleaner code, and manage complexity. The goal is to help you code quicker, make fewer mistakes, and keep your solutions running fast.

Typing matters. The faster you type, the more time you save. Accuracy also counts—mistakes slow you down. Next, we cut down on code size without losing what’s important. Using tools like the Standard Template Library (STL), you can write less code and keep it clean. This is about direct, simple code that does the job right.

Complexity is a problem that needs control. Good algorithms are fast and use memory well. Managing complexity ensures your code works when input sizes grow. The advice here is built for contests: code for speed, code for one-time use. It is not for long-term projects where readability and maintenance matter more.

## 2.1. Typing Tips

If you don’t type quickly, **you should invest at least two hours per week** on the website: [https://www.speedcoder.net](https://www.speedcoder.net). Once you have completed the basic course, select the C++ lessons and practice regularly. Time is crucial in competitive programming, and slow typing can be disastrous.

To expand on this, efficient typing isn’t just about speed; it’s about reducing errors and maintaining a steady flow of code. When you're in a competitive programming, every second matters. Correcting frequent typos or having to look at your keyboard will significantly slow down your progress. Touch typing—knowing the layout of the keyboard and typing without looking—becomes a vital skill.

In a typical competitive programming contest, you have to solve several, typically 12 or 15, problems within a set time, about five hours. Typing fast lets you spend more time solving problems instead of struggling to get the code in. But speed means nothing without accuracy. Accurate and fast typing ensures that once you know the solution, you can code it quickly and correctly.

Typing slow or making frequent errors costs you valuable time. You waste time fixing mistakes, lose focus on solving the problem, and increase the chance of not finishing on time. Aim for a typing speed of at least 60 words per minute with high accuracy. Websites like [https://www.speedcoder.net](https://www.speedcoder.net) let you practice typing code syntax, which helps more than regular typing lessons. Learning specific shortcuts in C++ or Python boosts your speed in real coding situations.

$$
\text{Time spent fixing errors} + \text{Time lost from slow typing} = \text{Lower overall performance in competitive programming}
$$

To improve your typing in competitive programming, start by using IDE shortcuts. Learn the keyboard shortcuts for your preferred Integrated Development Environment (IDE). Shortcuts help you navigate and edit code faster, cutting down the time spent moving between the keyboard and mouse. In [ICPC](https://icpc.global/) contests, the IDE is usually [Eclipse](https://www.eclipse.org/downloads/packages/release/helios/sr2/eclipse-ide-cc-developers) or [VsCode](https://code.visualstudio.com/), so knowing its shortcuts can boost your efficiency. Always check which IDE will be used in each competition since this may vary. And use it daily while training.

while typing, focus on frequent patterns in your code. Practice typing common elements like loops, if-else conditions, and function declarations. Embedding these patterns into your muscle memory saves time during contests. The faster you can type these basic structures, the quicker you can move on to solving the actual problem.

To succeed in a C++ programming competition, your first challenge is to type the following code in under two minutes. If you can't, don't give up. Just keep practicing. To be the best in the world at anything, no matter what it is, the only secret is to train and train some more.

```cpp
#include <iostream>
#include <vector>
#include <span>
#include <string>
#include <algorithm>
#include <random>

// Type aliases
using VI = std::vector<int>;
using IS = std::span<int>;
using STR = std::string;

// Function to double each element in the vector
void processVector(VI& vec) {
    std::cout << "Processing vector...\n";
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] *= 2;
    }
}

// Function to display elements of a span
void displaySpan(IS sp) {
    std::cout << "Displaying span: ";
    for (const auto& elem : sp) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    VI numbers;
    STR input;

    // Input loop: collect numbers from user
    std::cout << "Enter integers (type 'done' when finished):\n";
    while (true) {
        std::cin >> input;
        if (input == "done") break;
        numbers.push_back(std::stoi(input));
    }

    // Process and display the vector
    processVector(numbers);
    std::cout << "Processed vector:\n";
    int index = 0;
    while (index < numbers.size()) {
        std::cout << numbers[index] << " ";
        ++index;
    }
    std::cout << "\n";

    // Shuffle the vector using a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(numbers.begin(), numbers.end(), gen);

    std::cout << "Shuffled vector:\n";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << "\n";

    // Display a span of the first 5 elements (if available)
    if (numbers.size() >= 5) {
        IS numberSpan(numbers.data(), 5);
        displaySpan(numberSpan);
    }

    // Calculate sum of elements at even indices
    int sum = 0;
    for (int i = 0; i < numbers.size(); i += 2) {
        sum += numbers[i];
    }
    std::cout << "Sum of elements at even indices: " << sum << "\n";

    return 0;
}
```

Don't give up before trying. If you can't type fast enough, don't stop here. Keep going. With each new algorithm, copy and practice again until it feels natural to type between 60 and 80 words per minute with an accuracy above $95%$.

## 2.2. Typing Less in Competitive Programming

In competitive programming, time is a critical resource and C++ is a language where you have to type a lot. Therefore, optimizing typing speed and avoiding repetitive code can make a significant difference. Below, we will discuss strategies to minimize typing when working with `std::vector` during competitive programmings, where access to the internet or pre-prepared code snippets may be restricted.

### 2.1.2. Abbreviations

We can use `#define` to create short aliases for common vector types. This is particularly useful when you need to declare multiple vectors throughout the code.

```cpp
#define VI std::vector<int>
#define VVI std::vector<std::vector<int>>
#define VS std::vector<std::string>
```

With these definitions, declaring vectors becomes much faster:

```cpp
VI numbers;  // std::vector<int> numbers;
VVI matrix;  // std::vector<std::vector<int>> matrix;
VS words;    // std::vector<std::string> words;
```

In C++, you can use `#define` to create macros and short aliases. Macros can define constants or functions at the preprocessor level. But in C++20, most macros have better alternatives. A typical use is reduce the definition and common operations with vectors. Some examples follow:

```cpp
#define VI std::vector<int>
#define VVI std::vector<std::vector<int>>
#define VS std::vector<std::string>
```

With these definitions, declaring vectors becomes much faster:

```cpp
VI numbers;  // std::vector<int> numbers;
VVI matrix;  // std::vector<std::vector<int>> matrix;
VS words;    // std::vector<std::string> words;
```

_Macros can cause problems_. They ignore scopes and can lead to unexpected behavior. For constants, use `constexpr`, `const` or `` instead of macros.

```cpp
// Old way using macros
#define PI 3.14159

// Modern way using constexpr
constexpr double PI = 3.14159;
```

> The `constexpr` keyword tells the compiler to evaluate a function or variable at compile time. If possible, the compiler computes it before runtime. This helps in optimization, as constants are determined during compilation, not execution.
>
> When you use `constexpr` with variables, the compiler knows the value at compile time. The value is constant and cannot change.
>
> ```cpp
> constexpr int max_items = 42;  // Value known at compile time, cannot change.
> ```
>
> You can use `constexpr` variables to define array sizes or template parameters when their values are known before or during compilation.
>
> ```cpp
> constexpr int size = 10;
> int array[size];  // Valid, size is a constant expression.
> ```
>
> We can use `constexpr` with Functions. A `constexpr` function can be evaluated at compile time if its inputs are constant expressions. All operations inside must be valid at compile time.
>
> ```cpp
> constexpr int factorial(int n) {
>     return n <= 1 ? 1 : n * factorial(n - 1);  // Recursive function computed at compile time.
> }
> ```
>
> If you call `factorial(6)` with a constant value, the compiler computes it at compile time and replaces the call with `720`.
>
> `constexpr` is useful in many contexts. It helps construct constant data, optimize algorithms, and define compile-time logic. Here are some examples:
>
> 1. Compile-time array size:
>
>    ```cpp
>    constexpr int size = 5;
>    int array[size];  // Size computed at compile time.
>    ```
>
> 2. Compile-time strings:
>
>    ```cpp
>    constexpr const char* greet() { return "Hello, World!"; }
>    constexpr const char* message = greet();  // Message computed at compile time.
>    ```
>
> 3. Compile-time mathematical operations:
>
>    ```cpp
>    constexpr int area(int length, int width) {
>        return length * width;
>    }
>    constexpr int room_area = area(10, 12);  // Computed at compile time.
>    ```
>
> In competitive programming, `constexpr` can be an advantage or a disadvantage. Advantage, `constexpr` can optimize code by computing results at compile time. This saves processing time during execution. If certain values are constant, you can precompute them with `constexpr`. Disadvantage: Many problems have dynamic input provided at runtime. `constexpr` cannot compute values that depend on runtime input. Since runtime efficiency is crucial, `constexpr` use is limited when inputs are dynamic.
>
> Overall, `constexpr` is valuable when dealing with static data or fixed input sizes. But in typical ICPC-style competitions, you use it less often because most problems require processing dynamic input.
>
> _For functions, macros can be unsafe_. They don't respect types or scopes. Modern C++ provides templates and `constexpr` functions.
>
> ```cpp
> // Macro function
> #define SQUARE(x) ((x) * (x))
>
> // Template function
> template<typename T>
> constexpr T square(T x) {
>     return x * x;
> }
> ```
>
> _Macros are processed before compilation. This makes debugging hard_. The compiler doesn't see macros the same way it sees code. With modern C++, you have better tools that the compiler understands. C++20 offers features like `constexpr` functions, inline variables, and templates. These replace most uses of macros. They provide type safety and respect scopes. They make code easier to read and maintain. You can define a `constexpr` function to compute the square of a number:
>
> ```cpp
> constexpr int square(int n) {
>     return n * n;
> }
> ```
>
> If you call `square(5)` in a context requiring a constant expression, the compiler evaluates it at compile time.
>
> In summary, avoid macros when you can. Use modern C++ features instead. They make your code safer and clearer.

**A better way to reduce typing time is by using `typedef` or `using` to create abbreviations for frequently used vector types.** In many cases, the use of `#define` can be replaced with more modern and safe C++ constructs like `using`, `typedef`, or `constexpr`. **The old `#define` does not respect scoping rules and does not offer type checking, which can lead to unintended behavior**. Using `typedef` or `using` provides better type safety and integrates smoothly with the C++ type system, making the code more predictable and easier to debug.

For example:

```cpp
#define VI std::vector<int>
#define VVI std::vector<std::vector<int>>
#define VS std::vector<std::string>
```

Can be replaced with `using` or `typedef` to create type aliases:

```cpp
using VI = std::vector<int>;
using VVI = std::vector<std::vector<int>>;
using VS = std::vector<std::string>;

// Or using typedef (more common in C++98/C++03)
typedef std::vector<int> VI;
typedef std::vector<std::vector<int>> VVI;
typedef std::vector<std::string> VS;
```

**`using` and `typedef` are preferred because they respect C++ scoping rules and offer better support for debugging, making the code more secure and readable**. nevertheless, there are moments when we need a constant function.

If you have macros that perform calculations, you can replace them with `constexpr` functions:

```cpp
  #define SQUARE(x) ((x) * (x))
```

Can be replaced with:

```cpp
 constexpr int square(int x) {
     return x * x;
}
```

**`constexpr` functions provide type safety, avoid unexpected side effects, and allow the compiler to evaluate the expression at compile-time, resulting in more efficient and safer code**.

In competitive programming, you might think using `#define` is the fastest way to type less and code faster. But `typedef` or `using` are usually more efficient. They avoid issues with macros and integrate better with the compiler. **While reducing variable names or abbreviating functions might save time in a contest, remember that in professional code, clarity and maintainability are more important than typing speed**. So avoid using shortened names and unsafe constructs like `#define` in production code, libraries, or larger projects.

> To understand better.In C++, you can create aliases for types. This makes your code cleaner. You use `typedef` or `using` to do this.
>
> `typedef` lets you give a new name to an existing type.
>
> ```cpp
> using ull = unsigned long long;
> ```
>
> Now, `ull` is an alias for `unsigned long long`. You can use it like this:
>
> ```cpp
> ull bigNum = 123456789012345ULL;
> ```
>
> In C++, numbers need type-specific suffixes like `ULL`. When you write `ull bigNumber = 123456789012345ULL;`, the `ULL` tells the compiler the number is an `unsigned long long`. Without it, the compiler might assume a smaller type like `int` or `long`, which can't handle large values. This leads to errors and bugs. The suffix forces the right type, avoiding overflow and keeping your numbers safe. It’s a simple step but crucial. The right suffix means the right size, no surprises.
>
> **In C++, suffixes are also used with floating-point numbers to specify their exact type**. The suffix `f` designates a `float`, while no suffix indicates a `double`, and `l` or `L` indicates a `long double`. By default, the compiler assumes `double` if no suffix is provided. Using these suffixes is important when you need specific control over the type, such as saving memory with `float` or gaining extra precision with `long double`. The suffix ensures that the number is treated correctly according to your needs.Exact type, exact behavior.

### 2.2.2. Predefining Common Operations

If you know that certain operations, such as sorting or summing elements, are frequent in a competitive programming, consider defining these operations at the beginning of the code. _The only real reason to use a macro in competitive programming is to predefine functions_. For example:

```cpp
// A function to sort vectors
#define SORT_VECTOR(vec) std::sort(vec.begin(), vec.end())
```

This function can be further simplified in its use:

```cpp
#include <vector>
#include <algorithm>

#define SORT_VECTOR(vec) std::sort(vec.begin(), vec.end())
#define ALL(vec) vec.begin(), vec.end()
using VI = std::vector<int>; // Alias with using needs a semicolon

// Usage:
VI vec = {5, 3, 8, 1};
SORT_VECTOR(vec); // Sorts the vector using the SORT_VECTOR macro
// Alternatively, you can sort using ALL macro:
std::sort(ALL(vec)); Another way to use the macro to sort
```

> In C++, `#include` brings code from libraries into your program. It lets you use functions and classes defined elsewhere. The line `#include <vector>` includes the vector library.
>
> Vectors are dynamic arrays. They can change size at runtime. You can add or remove elements as needed. We will know more about vectors and the vector library soon. In early code fragments we saw some examples of vector initialization.
>
> The line `#include <algorithm>` includes the algorithm library. It provides functions to work with data structures. You can sort, search, and manipulate collections.
>
> We can merge `<vector>` and `<algorithm>` for efficient data processing. We’ve seen this in previous code examples where we used competitive programming techniques. Without competitive programming tricks and hacks, the libraries can be combined like this:
>
> ```cpp
> #include <vector>
> #include <algorithm>
> #include <iostream>
>
> int main() {
>     std::vector<int> numbers = {4, 2, 5, 1, 3};
>     std::sort(numbers.begin(), numbers.end());
>
>     for (int num : numbers) {
>         std::cout << num << " ";
>     }
>     return 0;
> }
> ```
>
> This program, a simple example of professional code, sorts the numbers and prints:
>
> ```txt
> 1 2 3 4 5
> ```

**Summing Elements**:

```cpp
int sum_vector(const VI& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0);
}
```

To calculate the sum of a vector's elements:

```cpp
int total = sum_vector(numbers);
```

### 2.3.2. Using Lambda Functions

Starting with C++11, C++ introduced lambda functions. Lambdas are anonymous functions that can be defined exactly where they are needed. If your code needs a simple function used only once, you should consider using lambdas. Let’s start with a simple example, written without competitive programming tricks.

```cpp
#include <iostream>  // Includes the input/output stream library for console operations
#include <vector>    // Includes the vector library for using dynamic arrays
#include <algorithm> // Includes the algorithm library for functions like sort

// Traditional function to sort in descending order
bool compare(int a, int b) {
    return a > b; // Returns true if 'a' is greater than 'b', ensuring descending order
}

int main() {
    std::vector<int> numbers = {1, 3, 2, 5, 4}; // Initializes a vector of integers

    // Uses the compare function to sort the vector in descending order
    std::sort(numbers.begin(), numbers.end(), compare);

    // Prints the sorted vector
    for (int num : numbers) {
        std::cout << num << " "; // Prints each number followed by a space
    }
    std::cout << "\n"; // Prints a newline at the end

    return 0; // Returns 0, indicating successful execution
}
```

This code sorts a vector of integers in descending order using a traditional comparison function. It begins by including the necessary libraries: `<iostream>` for input and output, `<vector>` for dynamic arrays, and `<algorithm>` for sorting operations. The `compare` function is defined to take two integers, returning `true` if the first integer is greater than the second, setting the sorting order to descending.

In the `main` function, a vector named `numbers` is initialized with the integers `{1, 3, 2, 5, 4}`. The `std::sort` function is called on this vector, using the `compare` function to sort the elements from highest to lowest. After sorting, a `for` loop iterates through the vector, printing each number followed by a space. The program ends with a newline to cleanly finish the output. This code is a simple and direct example of using a custom function to sort data in C++. Now, let's see the same code using lambda functions and other competitive programming tricks.

```cpp
#include <iostream>  // Includes the input/output stream library for console operations
#include <vector>    // Includes the vector library for using dynamic arrays
#include <algorithm> // Includes the algorithm library for functions like sort

#define ALL(vec) vec.begin(), vec.end() // Macro to simplify passing the entire range of a vector
using VI = std::vector<int>; // Alias for vector<int> to simplify code and improve readability

int main() {
    VI num = {1, 3, 2, 5, 4}; // Initializes a vector of integers using the alias VI

    // Sorts the vector in descending order using a lambda function
    std::sort(ALL(num), [](int a, int b) { return a > b; });

    // Prints the sorted vector
    for (int n : num) {
        std::cout << n << " "; // Prints each number followed by a space
    }
    std::cout << "\n"; // Prints a newline at the end

    return 0; // Returns 0, indicating successful execution
}
```

To see the typing time gain, just compare the normal definition of the `compare` function followed by its usage with the use of the lambda function.

This code sorts a vector of integers in descending order using a lambda function, a modern and concise way to define operations directly in the place where they are needed. It starts by including the standard libraries for input/output, dynamic arrays, and algorithms. The macro `ALL(vec)` is defined to simplify the use of `vec.begin(), vec.end()`, making the code cleaner and shorter.

An alias `VI` is used for `std::vector<int>`, reducing the verbosity when declaring vectors. Inside the `main` function, a vector named `num` is initialized with the integers `{1, 3, 2, 5, 4}`. The `std::sort` function is called to sort the vector, using a lambda function `[](int a, int b) { return a > b; }` that sorts the elements in descending order.

The lambda is defined and used inline, removing the need to declare a separate function like `compare`. After sorting, a `for` loop prints each number followed by a space, ending with a newline. This approach saves time and keeps the code concise, highlighting the effectiveness of lambda functions in simplifying tasks that would otherwise require traditional function definitions.

> Lambda functions in C++, introduced in C++11, are anonymous and defined where they are needed. They shine in short, temporary tasks like inline calculations or _callbacks_. Unlike regular functions, lambdas can capture variables from their surrounding scope. With C++20, lambdas became even more powerful and flexible, extending their capabilities beyond simple operations.
>
> The general syntax for a lambda function in C++ is as follows:
>
> ```cpp
> [capture](parameters) -> return_type { // function body};
> ```
>
> Where:
>
> - `Capture`: Specifies which variables from the surrounding scope can be used inside the lambda. Variables can be captured by value `[=]` or by reference `[&]`. You can also specify individual variables, such as `[x]` or `[&x]`, to capture them by value or reference, respectively.
>   `Parameters`: The input parameters for the lambda function, similar to function arguments.
>   `Return Type`: Optional in most cases, as C++ can infer the return type automatically. However, if the return type is ambiguous or complex, it can be specified explicitly using `-> return_type`.
> - `Body`: The actual code to be executed when the lambda is called.
>
> C++20 brought new powers to lambdas. Now, they work in **immediate functions** with `consteval`, running faster at compile-time. Lambdas can be default-constructed without capturing anything. They can also use **template parameters**, making them more flexible and generic. Let’s see some examples.
>
> Example 1: Basic Lambda Function: A simple example of a lambda function that sums two numbers:
>
> ```cpp
> auto sum = [](int a, int b) -> int {return a + b;};
> std::cout << sum(5, 3);  // Outputs: 8
> ```
>
> Example 2: Lambda with Capture: In this example, a variable from the surrounding scope is captured by value:
>
> ```cpp
>  int x = 10; // Initializes an integer variable x with the value 10
>
>  // Defines a lambda function that captures x by value (creates a copy of x)
>  auto multiply = [x](int a) {return x * a;}; // Multiplies the captured value of x by the argument a
>
>  std::cout << multiply(5);  // Calls the lambda with 5; Outputs: 50
> ```
>
> Here, the lambda captures `x` by value and uses it in its body. This means `x` is copied when the lambda is created. The lambda holds its own version of `x`, separate from the original. Changes to `x` outside the lambda won’t affect the copy inside. It’s like taking a snapshot of `x` at that moment. The lambda works with this snapshot, keeping the original safe and unchanged. But this copy comes at a cost—extra time and memory are needed. For simple types like integers, it’s minor, but for larger objects, the overhead can add up.
>
> Example 3: Lambda with Capture by Reference: In this case, the variable `y` is captured by reference, allowing the lambda to modify it:
>
> ```cpp
> int y = 20;  // Initializes an integer variable y with the value 20
>
> // Defines a lambda function that captures y by reference (no copy is made)
> auto increment = [&y]() {
>     y++;  // Increments y directly
> };
>
> increment();  // Calls the lambda, which increments y
> std::cout << y;  // Outputs: 21
>
> ```
>
> In this fragment, there’s no extra memory or time cost. The lambda captures `y` by reference, meaning it uses the original variable directly. No copy is made, so there’s no overhead. When `increment()` is called, it changes `y` right where it lives. The lambda works with the real `y`, not a snapshot, so any change happens instantly and without extra resources. This approach keeps the code fast and efficient, avoiding the pitfalls of capturing by value. The result is immediate and uses only what’s needed. **In competitive or high-performance programming, we capture by reference. It's faster and uses less memory**.
>
> Example 4: Generic Lambda Function with C++20: With C++20, lambdas can now use template parameters, making them more generic:
>
> ```cpp
> // Defines a generic lambda function using a template parameter <typename T>
> // The lambda takes two parameters of the same type T and returns their sum
> auto generic_lambda = []<typename T>(T a, T b) { return a + b; };
>
> std::cout << generic_lambda(5, 3);      // Calls the lambda with integers, Outputs: 8
> std::cout << generic_lambda(2.5, 1.5);  // Calls the lambda with doubles, Outputs: 4.0
> ```
>
> This code defines a generic lambda using a template parameter, a feature from C++20. The lambda accepts two inputs of the same type `T` and returns their sum. It’s flexible—first, it adds integers, then it adds doubles. The power of this lambda is in its simplicity and versatility. It’s short, clear, and works with any type as long as the operation makes sense. C++20 lets you keep your code clean and adaptable, making lambdas more powerful than ever. And it doesn’t stop there.
>
> Default-constructed lambdas: In C++20, lambdas that don’t capture variables can be default-constructed. You can create them, assign them, and save them for later without calling them right away. This makes it easy to store and pass lambdas when you need a default behavior.
>
> ```cpp
> #include <iostream>
> #include <vector>
> #include <algorithm>
>
> #define ALL(vec) vec.begin(), vec.end() // Macro to simplify passing the entire range of a vector
> using VI = std::vector<int>; // Alias for vector<int> to simplify code and improve readability
>
> // Define a default-constructed lambda that prints a message
> auto print_message = []() {std::cout << "Default behavior: Printing message." << "\n";};
>
> int main() {
>     // Store the default-constructed lambda and call it later
>     print_message();
>
>     // Define a vector and use the lambda as a fallback action
>     VI num = {1, 2, 3, 4, 5};
>
>     // If vector is not empty, do something; else, call the default lambda
>     if (!num.empty()) {
>         std::for_each(ALL(num), [](int n) {std::cout << n * 2 << " ";});  // Prints double of each number
>     } else {
>         print_message(); // Calls the default lambda if no numbers to process
>     }
>
> return 0;
> }
> ```
>
> **This feature lets you set up lambdas for later use (deferred execution)**. In the last code, the lambda `print_message` is default-constructed. It captures nothing and waits until it’s needed. The main function shows this in action. If the vector has numbers, it doubles them. If not, it calls the default lambda and prints a message. C++20 makes lambdas simple and ready for action, whenever you need them.
>
> Immediate lambdas: C++20 brings in `consteval`, a keyword that forces functions to run at compile-time. With lambdas, this means the code is executed during compilation, and the result is set before the program starts. When a lambda is used in a `consteval` function, it must run at compile-time, making your code faster and results predictable.
>
> **In programming competitions, `consteval` lambdas are rarely useful**. Contests focus on runtime performance, not compile-time tricks. Compile-time evaluation doesn’t give you an edge when speed at runtime is what counts. Most problems don’t benefit from execution before the program runs; the goal is to be fast during execution.
>
> `Consteval` ensures the function runs only at compile-time. If you try to use a `consteval` function where it can’t run at compile-time, you’ll get a compile-time error. It’s strict: no runtime allowed.
>
> ```cpp
> consteval auto square(int x) {
>     return [] (int y) { return y * y; }(x);
> }
> int value = square(5);  // Computed at compile-time
> ```
>
> In this example, the lambda inside the `square` function is evaluated at compile-time, producing the result before the program starts execution. **Programming contests focus on runtime behavior and dynamic inputs, making `consteval` mostly useless**. In contests, you deal with inputs after the program starts running, so compile-time operations don’t help. The challenge is to be fast when the program is live, not before it runs.
>
> Finally, we have template lambdas. C++20 lets lambdas take template parameters, making them generic. They can handle different data types without needing overloads or separate template functions. The template parameter is declared right in the lambda’s definition, allowing one lambda to adapt to any type.
>
> Example:
>
> ```cpp
> auto generic_lambda = []<typename T>(T a, T b) {
>     return a + b;
> };
> std::cout << generic_lambda(5, 3);      // Outputs: 8
> std::cout << generic_lambda(2.5, 1.5);  // Outputs: 4.0
> ```
>
> **Template lambdas are a powerful tool in competitive programming**. They let you write one lambda that works with different data types, saving you time and code. Instead of writing multiple functions for integers, doubles, or custom types, you use a single template lambda. It adapts on the fly, making your code clean and versatile. In contests, where every second counts, this can speed up coding and reduce bugs. You get generic, reusable code without the hassle of writing overloads or separate templates.
>
> _Lambdas are great for quick, one-time tasks. But too many, especially complex ones, can make code harder to read and maintain. In competitive programming, speed often trumps clarity, so this might not seem like a big deal. Still, keeping code readable helps, especially when debugging tough algorithms. Use lambdas wisely._

## 2.3. Optimizing File I/O in C++ for competitive programmings

In competitive programming contests, especially with large datasets, programs often need to read input from big files.

In C++, file input and output (I/O) operations are managed using classes from the `<fstream>` library. The main classes are `std::ifstream`, `std::ofstream`, and `std::fstream`. These classes serve different purposes: reading, writing, and both reading and writing.

- `std::ifstream`: Used for reading from files.
- `std::ofstream`: Used for writing to files.
- `std::fstream`: Used for both reading and writing to files.

The `std::ifstream` class reads files. It is only for input. It inherits from `std::istream`, the main class for input in C++. Use `std::ifstream` to open a file and read its data. You can read line by line or in parts. It is straightforward and efficient. This class also checks file status and handles errors. It is a basic tool for reading in C++.

In our code, use `std::ifstream` to open a text file and read its contents:

```cpp
std::ifstream file("path_to_file");
```

The line `std::ifstream file("path_to_file");` opens a file. It uses the file name given `path_to_file`. If the file does not open, the stream becomes invalid. Off course, you can use `std::ifstream` to read files directly from the command line.

```cpp
#include <fstream>  // Includes the library for file input/output operations
#include <iostream> // Includes the library for input/output operations on the console
#include <string>   // Includes the string library for handling strings

int main(int argc, char* argv[]) {
    // Checks if the program received exactly one argument (filename) besides the program name
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n"; // Prints an error message with the correct usage
        return 1; // Returns 1 to indicate an error occurred
    }

    // Opens the file specified by the command-line argument for reading
    std::ifstream file(argv[1]);

    // Checks if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << argv[1] << "\n"; // Prints an error message if the file couldn't be opened
        return 1; // Returns 1 to indicate an error occurred
    }

    std::string line; // Declares a string variable to store each line of the file

    // Reads the file line by line and outputs each line to the console
    std::cout << "Contents of the file:\n";
    while (std::getline(file, line)) { // Reads each line from the file into the 'line' variable
        std::cout << line << "\n"; // Prints the current line to the console
    }

    file.close(); // Closes the file after reading
    return 0; // Returns 0 to indicate successful execution
}
```

The program starts by checking the number of command-line arguments:

```cpp
if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
    return 1;
}
```

> In C++, `argc` and `argv` are used to handle command-line arguments. They are part of the parameters for the main function:
>
> ```cpp
> int main(int argc, char* argv[])
> ```
>
> `argc` stands for "argument count". It tells you how many command-line arguments were passed to the program. The count includes the program's name itself, so `argc` is always at least $1$.
>
> `argv` stands for "argument vector". It is an array of pointers to strings. Each string is a command-line argument. The first element, `argv[0]`, is always the program’s name. The other elements, `argv[1]`, `argv[2]`, and so on, are the arguments given by the user.
>
> When you run a program from the command line, you can pass additional data right after the program’s name. For example:
>
> ```shell
> ./program file.txt
> ```
>
> Here, `argc` will be $2$. `argv[0]` is `./program`, and `argv[1]` is file.txt.
>
> The command line is read by the operating system before your program starts. The system stores each space-separated word as a string. The program reads this input using `argc` and `argv`. By checking `argc`, you can make sure the user provided the correct number of arguments. `argv` lets you access and use those arguments directly inside your code.
>
> It’s a simple way to pass information when starting a program, especially useful for filenames, options, or parameters that the program needs to operate correctly.

It expects one argument besides the program name. If there is no filename, it prints a usage message and stops. Next, the program tries to open the file:

```cpp
std::ifstream file(argv[1]);
```

The line `std::ifstream file(argv[1]);` creates a file stream named `file`. It tries to open the file specified by `argv[1]`, which is the first command-line argument given by the user.

> A file stream is a connection between your program and a file. It lets the program read from or write to the file. In C++, file streams come from the `<fstream>` library. They act like a bridge, taking data from the program to the file or bringing data from the file into the program. A file stream opens the file, manages it, and handles any errors that occur. It makes working with files simple and direct.
>
> If the file exists and can be opened, `file` is ready to read from that file. If it fails, `file` will be invalid, and nothing can be read. This line sets up a direct link between the program and the file, letting the program read the file’s content.

```cpp
if (!file.is_open()) {
    std::cerr << "Error: Could not open the file " << argv[1] << "\n";
    return 1;
}
```

`file.is_open()` checks if the file stream opened the file. If it’s open, it returns `true`. If not, it returns `false`. This helps you know if you can read or write.

> `std::cerr` prints error messages. It’s fast and shows messages right away. We also have `std::cout` prints regular output. It’s used to show results or messages to the user and `std::cin` reads input from the user. It takes what you type and gives it to the program.
>
> Use `std::cout` to show, `std::cin` to read, and `std::cerr` to warn.

```cpp
while (std::getline(file, line)) {
    std::cout << line << "\n";
}
```

`std::getline` reads one line at a time from the file and puts it in line. It starts reading at the beginning of the line and stops when it reaches the end. It looks for the newline character to know when the line ends. After reading, it moves to the next line and repeats. Each line is printed right away with `std::cout`. This loop keeps going until there are no more lines left in the file.

```cpp
file.close();
```

`file.close()` ends the link to the file. It tells the program you are finished with it. Closing keeps things tidy and safe. It frees resources and makes sure all data is saved. Always close the file when you’re done.

It then ends successfully, returning zero.

We didn’t use `std::ofstream` in the code, but it’s key to know. `std::ofstream` writes to files. It comes from `std::ostream`, which handles all output in C++.

To open a file for writing, you use it like `std::ifstream`:

```cpp
std::ofstream outFile("output.txt");
```

This line creates or opens output.txt for writing. If the file exists, it clears the old contents first. It’s ready to write new data right away.

`std::fstream` combines what `std::ifstream` and `std::ofstream` do. It lets you read from and write to the same file. It comes from `std::iostream`, which handles input and output both ways.

Here’s how you open a file for reading and writing:

```cpp
std::fstream file("data.txt", std::ios::in | std::ios::out);
```

This line opens `data.txt` for both reading and writing. The flags `std::ios::in | std::ios::out` tell the stream to allow both input and output. You can read and write without closing the file in between.
Or to use files read from command line we could use:

```cpp
#include <fstream>  // Includes the library for file input/output operations
#include <iostream> // Includes the library for console input/output
#include <string>   // Includes the string library for string manipulation

int main(int argc, char* argv[]) {
    // Checks if the correct number of arguments is provided.
    // The program expects exactly one argument besides the program name.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n"; // Prints a usage message to the error stream if arguments are incorrect.
        return 1; // Exits with an error code 1 indicating incorrect usage.
    }

    // Opens the file specified by the command-line argument for both reading and writing.
    // std::ios::in | std::ios::out allows input (reading) and output (writing).
    std::fstream file(argv[1], std::ios::in | std::ios::out);

    // Checks if the file was opened successfully.
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << argv[1] << "\n"; // Prints an error message if the file can't be opened.
        return 1; // Exits with an error code 1 indicating failure to open the file.
    }

    std::string line; // Declares a string variable to store each line read from the file.

    // Reads the file line by line and prints each line to the console.
    std::cout << "Contents of the file:\n";
    while (std::getline(file, line)) { // Reads each line from the file and stores it in the 'line' variable.
        std::cout << line << "\n"; // Prints the current line to the console.
    }

    // Resets the file stream state to clear any error flags (like EOF).
    file.clear(); // Clears any error flags that might have been set during reading.
    // Repositions the file pointer to the beginning of the file for further operations.
    file.seekg(0, std::ios::beg); // Moves the file's read position back to the start.

    // Writes a new line at the end of the file.
    file << "\nNew line added to the file.\n"; // Adds a new line to the file content.

    // Clears any error flags again and repositions the file pointer to the beginning.
    file.clear(); // Clears flags that may have been set after writing.
    file.seekg(0, std::ios::beg); // Moves the read position back to the beginning.

    std::cout << "\nUpdated contents of the file:\n";

    // Reads and prints the updated content of the file.
    while (std::getline(file, line)) { // Reads the file again after the new line is added.
        std::cout << line << "\n"; // Prints each updated line to the console.
    }

    file.close(); // Closes the file to release resources.
    return 0; // Returns 0 indicating successful execution.
}
```

The code opens a file for reading and writing. Here’s how it manages the reading and writing positions with the pointers. First, the program reads the file line by line:

```cpp
while (std::getline(file, line)) {
    std::cout << line << "\n";
}
```

This loop reads every line until the end. The pointer for reading (`seekg`) moves to the end of the file by the time the loop finishes. The pointer for writing (`seekp`) also ends up at the end because it follows the reading operations. This is why, when you write later, the text goes to the end. Next, the code resets the state and positions:

```cpp
file.clear(); // Clears any flags like EOF.
file.seekg(0, std::ios::beg); // Moves the read pointer to the start of the file.
```

`file.clear()` resets any error flags. `file.seekg(0, std::ios::beg)` moves only the reading pointer back to the beginning of the file. It doesn’t move the writing pointer. The write pointer remains at the end. Then, the program writes a new line:

```cpp
file << "\nNew line added to the file.\n";
```

This writes at the current write pointer position, which is still at the end. The read pointer is at the beginning, but it doesn’t affect where the writing happens. The writing pointer (`seekp`) hasn’t been moved, so it writes right where it was at the end.

Finally, the code resets the state again:

```cpp
file.clear(); // Clears flags after writing.
file.seekg(0, std::ios::beg); // Moves the read pointer back to the start again.
```

This prepares for reading the updated content from the beginning, but again, it only affects the read pointer. The write pointer is unaffected and remains where it finished writing.

In summary, the code reads to the end, clears state, and moves only the read pointer back. The write pointer stays at the end, which is why the new line is added at the end of the file, not somewhere else.

When opening files, you can set different modes using values from the `std::ios_base::openmode` enumeration. Each mode controls how the file is accessed.

`std::ios::in` opens the file for reading, which is the default for `std::ifstream`. `std::ios::out` opens it for writing, the default for `std::ofstream`. **`std::ios::app` opens the file for writing but always writes at the end without erasing the existing content**. `std::ios::ate` opens the file and moves the pointer directly to the end. `std::ios::trunc` opens the file and clears all its contents, starting fresh. `std::ios::binary` opens the file in binary mode, treating the file data as raw bytes.

### 2.3.1. Reading Lines More Efficiently\*\*

Using `std::getline()` works, but it can be slow when handling large files. Each call reads one line at a time, creating a lot of overhead. To speed this up, you can implement a custom buffer that reads multiple lines at once. By storing more data in one go, you reduce the number of I/O operations and make reading much faster. This approach minimizes the repetitive calls to the I/O functions, which is often the bottleneck in processing big text files.

```cpp
#include <iostream>   // Includes the standard input/output stream library
#include <fstream>    // Includes the file stream library
#include <vector>     // Includes the vector library for dynamic arrays
#include <string>     // Includes the string library for string manipulation
#include <sstream>    // Includes the string stream library to handle buffer splitting

int main(int argc, char* argv[]) {
    // Checks if the program received exactly one argument (filename) besides the program name
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n"; // Prints error message if arguments are incorrect
        return 1; // Returns 1 to indicate incorrect usage
    }

    // Opens the specified file in binary mode for reading
    std::ifstream file(argv[1], std::ios::in | std::ios::binary);
    // Checks if the file opened successfully
    if (!file) {
        std::cerr << "Error: Could not open the file " << argv[1] << "\n"; // Prints error message if file can't be opened
        return 1; // Returns 1 to indicate failure in opening the file
    }

    // Moves the file pointer to the end to determine the file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg(); // Gets the file size in bytes
    file.seekg(0, std::ios::beg);   // Resets the file pointer to the beginning

    // Allocates a buffer to hold the entire file contents
    std::vector<char> buffer(fileSize);
    // Reads the entire file into the buffer
    file.read(buffer.data(), fileSize);

    // Converts the buffer into a string for processing
    std::string data(buffer.begin(), buffer.end());

    // Creates a string stream to process lines efficiently from the buffer
    std::istringstream dataStream(data);
    std::string line;

    // Reads lines from the string stream without calling std::getline() on the file directly
    std::cout << "Contents of the file read using custom buffering:\n";
    while (std::getline(dataStream, line)) {
        std::cout << line << "\n"; // Prints each line to the console
    }

    file.close(); // Closes the file after reading
    return 0; // Returns 0 to indicate successful execution
}
```

Let's break the most important part of this code:

```cpp
std::vector<char> buffer(fileSize);
```

This line creates a buffer with `std::vector<char>`. The size matches `fileSize`, which is the total number of bytes in the file. The vector handles memory on its own, making it easy to fit the whole file. This buffer holds all the file’s data in one place.

```cpp
file.read(buffer.data(), fileSize);
```

`file.read()` reads the whole file in one go. `buffer.data()` points to the start of the buffer. `fileSize` tells it how many bytes to read. This reads everything at once, cutting down on repeated read calls. It makes the process fast because there’s no constant back and forth with the file.

```cpp
std::string data(buffer.begin(), buffer.end());
```

This line turns the buffer into a string. It takes data from `buffer.begin()` to `buffer.end()`. Now, the raw bytes are text, and you can work with them as a whole string. This makes it simple to process, split, or manipulate the data.

```cpp
std::istringstream dataStream(data);
std::string line;
```

The string is put into a `std::istringstream`. This stream treats the string like a file in memory. It reads lines quickly without touching the file again. The data stays in memory, ready to be read line by line without slow file I/O.

```cpp
std::cout << "Contents of the file read using custom buffering:\n";
while (std::getline(dataStream, line)) {
    std::cout << line << "\n"; // Prints each line to the console
}
```

This loop reads each line from the stream, not the file. `std::getline()` works on the string in memory, so it’s fast. No more file reads, just reading what’s already loaded. It prints each line, showing how efficient buffering and memory processing can be. This method keeps the file read quick and the processing smooth.

## 2.4. Competitive Programming and File I/O

There are faster ways to open and handle files in C++, especially when dealing with large data sets in competitive programming. These techniques can speed up file processing.

Use `std::ios::sync_with_stdio(false);` to disable the synchronization between C++ streams and C streams (`stdio` functions). This makes input and output faster because it removes the overhead of syncing with C-style input/output.

Turn off the synchronization with `cin.tie(nullptr);`. This disconnects `cin` from `cout`, so `cout` doesn’t flush every time `cin` is used. This can save time when reading and writing a lot of data.

Use larger buffers when reading and writing to minimize the number of operations. Reading a chunk of data at once, rather than line by line, can make your program faster.

Combine `std::ios::in | std::ios::out | std::ios::binary` when opening files to read and write in binary mode, reducing the time spent on formatting operations. These tweaks make your file operations lean and quick, perfect for big data tasks.

### 2.4.1. Use Manual Buffering

Manual buffering speeds up file handling by reading data in large chunks instead of line by line. This cuts down the overhead of repeated I/O operations. The code below shows how to read the whole file into a buffer efficiently, followed by an explanation of each step.

```cpp
#include <fstream>  // Includes the library for file input/output operations.
#include <iostream> // Includes the library for input/output operations on the console.
#include <vector>   // Includes the vector library for using dynamic arrays.

int main(int argc, char* argv[]) {
    // Checks if exactly one argument (the file name) is provided besides the program name.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file_name>\n"; // Prints usage instructions if incorrect arguments are provided.
        return 1; // Returns 1 to indicate an error in the number of arguments.
    }

    // Opens the file specified by the first command-line argument in binary mode for reading.
    std::ifstream file(argv[1], std::ios::in | std::ios::binary);
    // Checks if the file was opened successfully.
    if (!file) {
        std::cerr << "Error opening file: " << argv[1] << "\n"; // Prints an error message if the file cannot be opened.
        return 1; // Returns 1 to indicate failure in opening the file.
    }

    // Moves the file pointer to the end to determine the file size.
    file.seekg(0, std::ios::end); // Sets the file position to the end of the file.
    size_t fileSize = file.tellg(); // Uses tellg() to get the size of the file in bytes.
    file.seekg(0, std::ios::beg); // Moves the file pointer back to the beginning for reading.

    // Creates a buffer of the same size as the file to hold the file contents.
    std::vector<char> buffer(fileSize); // Allocates a vector of characters to store the entire file.

    // Reads the entire file into the buffer in one read operation.
    file.read(buffer.data(), fileSize); // Reads the file content into the buffer from start to end.

    // Processes the buffer contents.
    // Example: Prints the first 100 characters of the file or up to the file size if smaller.
    for (int i = 0; i < 100 && i < fileSize; ++i) {
        std::cout << buffer[i]; // Outputs each character to the console.
    }

    // Exits the program successfully.
    return 0;
}
```

Let’s break down the key lines that make this file reading efficient.

```cpp
file.seekg(0, std::ios::end);
```

This line moves the file pointer to the end. The function `seekg` sets where the next read starts. The $0$ means no offset, and `std::ios::end` moves the pointer straight to the end of the file. This step is crucial because it lets us find out the size of the file, which we need to create a buffer big enough to hold everything.

```cpp
size_t fileSize = file.tellg();
```

With the pointer at the end, `tellg()` gets the current position of the pointer, now at the file’s end. This position equals the total size of the file in bytes. We store this size in `fileSize`. Knowing this size allows us to set up a buffer that matches the file’s length exactly.

```cpp
file.seekg(0, std::ios::beg);
```

After we know the size, we move the pointer back to the start. `seekg(0, std::ios::beg)` places the pointer at the first byte of the file. Now, the file is ready to be read from the beginning.

```cpp
std::vector<char> buffer(fileSize);
```

Next, we create a buffer with `std::vector<char>` that’s as big as the file. This buffer holds the entire content of the file in memory. The vector automatically manages the memory, making it easier to handle large data. We access its data with `buffer.data()`.

```cpp
file.read(buffer.data(), fileSize);
```

Here, `file.read()` reads the whole file into the buffer. `buffer.data()` gives us the pointer to where the data will go. `fileSize` tells the program how many bytes to read. Since `fileSize` matches the file’s size, it reads everything in one go.

**Using `seekg()` to find the size and then reading everything at once cuts down on I/O operations. Instead of reading line by line or byte by byte, we grab all the data in a single action. This reduces system calls and slashes overhead, making it much faster, especially for large files.**

### 2.4.2 Using `mmap` for Faster File I/O in Unix-Based Systems

In competitive programming, especially in ICPC contests on Unix-based systems, every optimization matters. One effective technique is using `mmap`. It’s a fast way to handle large files by mapping them directly into memory. This allows almost instant access to file content, reducing the overhead of repeated reads.

`mmap` maps a file into your program’s memory. Once mapped, the file becomes part of the program’s memory space. You access its contents through pointers instead of read operations. It turns the file into a simple array in memory, removing the need for constant I/O calls.

This approach is highly effective in environments like ICPC, where files are large, and speed is critical. However, remember that `mmap` works only on Unix-based systems. It’s not portable and won’t run on Windows. Use it when efficiency is key, but know its limitations.

Here's an example of how you can use `mmap` to read a file efficiently in C++ on a Unix-based system:

```cpp
#include <sys/mman.h>  // Includes the library for memory mapping functions
#include <fcntl.h>     // Includes the library for file control options (like open)
#include <unistd.h>    // Includes the library for POSIX operating system API (like close)
#include <sys/stat.h>  // Includes the library for obtaining file status (like fstat)
#include <iostream>    // Includes the library for input/output operations

int main(int argc, char* argv[]) {
    // Checks if exactly one argument (the file name) is provided besides the program name
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file_name>\n"; // Prints usage instructions if the argument is missing
        return 1; // Returns 1 to indicate incorrect usage
    }

    // Open the file in read-only mode
    int fd = open(argv[1], O_RDONLY);
    // Checks if the file was opened successfully
    if (fd == -1) {
        std::cerr << "Error opening file: " << argv[1] << "\n"; // Prints an error message if the file cannot be opened
        return 1; // Returns 1 to indicate failure in opening the file
    }

    // Get the size of the file using fstat
    struct stat sb; // Declares a struct to hold the file status
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error getting file size\n"; // Prints an error message if unable to get file size
        close(fd); // Closes the file descriptor since it won't be used further
        return 1; // Returns 1 to indicate failure in getting the file status
    }
    size_t fileSize = sb.st_size; // Retrieves the file size from the stat struct

    // Memory-map the file into the process’s address space
    char* fileData = (char*)mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    // Checks if mmap failed to map the file
    if (fileData == MAP_FAILED) {
        std::cerr << "Error mapping file to memory\n"; // Prints an error message if mapping fails
        close(fd); // Closes the file descriptor since the mapping failed
        return 1; // Returns 1 to indicate failure in memory mapping
    }

    // Process the file data (example: print the first 100 characters)
    for (size_t i = 0; i < 100 && i < fileSize; ++i) {
        std::cout << fileData[i]; // Prints each character to the console up to the first 100 or file size
    }

    // Unmap the file from memory and close the file descriptor
    if (munmap(fileData, fileSize) == -1) {
        std::cerr << "Error unmapping file\n"; // Prints an error message if unmapping fails
    }
    close(fd); // Closes the file descriptor to release the resource

    return 0; // Returns 0 to indicate successful execution
}
```

The code we provided is not specific to C++20. It uses system calls and C libraries, not features of C++. Functions like `open`, `close`, `fstat`, and `mmap` are part of the Unix POSIX API, written in C. These work in C++ because C++ is compatible with C.

The libraries `<sys/mman.h>`, `<fcntl.h>`, `<unistd.h>`, and `<sys/stat.h>` are low-level C libraries used for file handling and memory mapping on Unix systems. They are not C++ libraries and are outside the C++ standard library.

In C++, especially in C++20, we have tools like `std::filesystem` that offer modern, safe ways to handle files. However, `mmap` and other direct system calls are not covered by the C++ standards. So while this code runs in C++, it does not use C++20 features. It’s C code running within a C++ program.

Let's try to understand what is happening in this file block by block.

```cpp
int fd = open(argv[1], O_RDONLY);
```

This line opens the file in read-only mode using `open()`. The `O_RDONLY` flag specifies that the file is opened for reading only, with no writing allowed. The function returns a file descriptor (`fd`), an integer that acts as a handle for the file, allowing the system to manage file operations. Other flags include `O_WRONLY` for write-only access and `O_RDWR` for both reading and writing. You can also use `O_CREAT` to create a file if it doesn’t exist, `O_TRUNC` to truncate an existing file, and `O_APPEND` to append data to the end of the file. The file descriptor is a key element in Unix-like systems, serving as a link between the program and the file, allowing efficient I/O operations.

```cpp
if (fd == -1) {
    std::cerr << "Error opening file: " << argv[1] << "\n";
    return 1;
}
```

This block checks if the file opened. If `fd` is $-1$, it means opening failed. The reasons can be the file doesn’t exist or lacks permissions. If it fails, it prints an error and exits with code $1$.

```cpp
struct stat sb;
if (fstat(fd, &sb) == -1) {
    std::cerr << "Error getting file size\n";
    close(fd);
    return 1;
}
size_t fileSize = sb.st_size;
```

The `stat` struct holds information about a file. When we call `fstat()`, it fills the `stat` struct with details about the file. This includes the file size, which we get from `sb.st_size`. The `stat` struct also stores other important details, like when the file was last modified, its permissions, and its type. But here, we only use it to get the size. If `fstat()` fails, it means something went wrong when trying to get the file’s info, so we print an error and close the file. The size stored in `fileSize` is crucial because it tells us how much memory we need to map the file.

```cpp
char* fileData = (char*)mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
```

This line maps the file into memory with `mmap()`. We use `nullptr` as the first argument, letting the system choose where to place the file in memory. `nullptr` means the pointer is empty; it’s a safe way to say we don’t care where it goes.

`fileSize` tells `mmap()` how much of the file to map, starting from the beginning. `PROT_READ` sets the memory as read-only. You can use `PROT_WRITE` to make it writable, `PROT_EXEC` to make it executable, or `PROT_NONE` to block access.

`MAP_PRIVATE` means any changes you make won’t affect the file. You could use `MAP_SHARED` if you want changes to be saved back to the file, or `MAP_ANONYMOUS` if you want to map memory without a file.

`fileData` becomes a pointer to the file’s data in memory. It acts like an array, making the file easy to read without constant I/O operations. You handle the file’s data directly in memory, which speeds up access and keeps things simple.

```cpp
if (fileData == MAP_FAILED) {
    std::cerr << "Error mapping file to memory\n";
    close(fd);
    return 1;
}
```

This block checks if mapping worked. If `fileData` equals `MAP_FAILED`, mapping failed, likely due to lack of permissions or memory. If it fails, it prints an error and closes the file.

```cpp
    for (size_t i = 0; i < 100 && i < fileSize; ++i) {
    std::cout << fileData[i];
}
```

This loop reads the first $100$ characters from the mapped memory. If the file is smaller, it stops at the file’s end. This is fast because it accesses the file directly in memory without extra read calls.

```cpp
if (munmap(fileData, fileSize) == -1) {
    std::cerr << "Error unmapping file\n";
}
close(fd);
```

This unmaps the file from memory with `munmap()`, freeing the memory used by the mapping. If it fails, it prints an error. The file descriptor is closed to free the resource, ensuring no leaks. This cleanup keeps the program tidy and avoids using system resources longer than needed.

So, `mmap` offers big advantages when handling large files. It’s fast because it maps the file directly into memory, cutting out repeated system calls. This reduces overhead and speeds up access. It’s simple too. Once mapped, the file acts like an array, making it easy to work with. It’s also efficient with memory. `mmap` loads only the parts of the file you need, instead of pulling the whole file into a buffer. This is a big win when dealing with large files.

Remember, `mmap` works only on POSIX systems like Linux, macOS, and other Unix-like setups. It’s not built into Windows, which can limit where your code runs. If you need your program to work on Windows too, consider alternatives or libraries that mimic `mmap` on different platforms. In programming contests like ICPC, where the environment is controlled and often Linux, `mmap` is a good choice. But if you need your code to run everywhere, use more universal methods like `std::ifstream` or `fread`, which work across all major operating systems.

**When using `mmap`, it’s crucial to manage resources properly. Always ensure you call `munmap()` to unmap the file when you’re done. Failing to unmap can lead to memory leaks, as the mapped memory remains reserved until the program ends. Use `munmap()` with the correct pointer and size to free up memory. Also, make sure to handle errors from `munmap()` gracefully, just as you would with `mmap()`. Proper cleanup keeps your program stable and avoids wasting system resources.**

### 2.4.3 Using Asynchronous I/O with `std::future` and `std::async` in C++20

In competitive programming, every millisecond counts. While `mmap` is a powerful tool for direct memory access, it's limited to Unix-based systems and specific use cases. For more flexibility and non-blocking operations, asynchronous I/O with `std::future` and `std::async` C++20 offers an effective alternative. This approach allows your program to continue running while waiting for I/O operations to complete, improving overall performance.

Asynchronous I/O separates file reading and writing tasks from the main execution flow. Instead of waiting for an operation to finish before moving on, the program can keep running, doing other work. This is especially useful when handling large files or when the program performs multiple I/O operations simultaneously. By offloading I/O tasks, you reduce idle time and make your program more responsive.

There are two classes to study now: `std::future` and `std::async`.

`std::future` and `std::async` are part of the C++ Standard Library’s support for concurrency. `std::async` launches a function asynchronously, usually in a separate thread, while `std::future` is used to retrieve the result once the function completes. For I/O tasks, you can use these tools to read from or write to files in the background, keeping your main thread free for other tasks.

Here’s an example of how asynchronous I/O works in practice:

```cpp
#include <iostream>    // Includes the standard input/output stream library
#include <fstream>     // Includes the file stream library for file handling
#include <future>      // Includes the library for asynchronous operations using std::future and std::async
#include <string>      // Includes the string library for string manipulation
#include <vector>      // Includes the vector library (not used directly here but commonly for dynamic arrays)

// Function to read a file asynchronously
std::string readFileAsync(const std::string& filename) {
    // Opens the file in input mode
    std::ifstream file(filename, std::ios::in);
    // Checks if the file opened successfully
    if (!file.is_open()) {
        // Throws an exception if the file cannot be opened
        throw std::runtime_error("Error opening file: " + filename);
    }

    // Reads the entire file content into a string using stream iterators
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content; // Returns the content of the file as a string
}

int main(int argc, char* argv[]) {
    // Checks if exactly one argument (the filename) is provided besides the program name
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n"; // Prints usage instructions if the argument is missing
        return 1; // Returns 1 to indicate incorrect usage
    }

    // Launch readFileAsync asynchronously, starting it in a separate thread
    std::future<std::string> result = std::async(std::launch::async, readFileAsync, argv[1]);

    // Main thread continues to run, doing other work without waiting for the file read to complete
    std::cout << "File is being read asynchronously...\n";

    // Retrieve the result once the async task completes
    try {
        // Blocks if necessary until the async task finishes and then retrieves the file content
        std::string content = result.get();
        // Prints the first 100 characters of the file content to the console
        std::cout << "File content:\n" << content.substr(0, 100) << "...\n";
    } catch (const std::exception& e) {
        // Catches and prints any errors that occurred during the async operation
        std::cerr << "Error: " << e.what() << '\n';
    }

    return 0; // Returns 0 to indicate successful execution
}
```

Breaking down the code again.

```cpp
std::string readFileAsync(const std::string& filename) {
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content;
}
```

`readFileAsync` is a function that reads the entire file into a string. It opens the file and uses `std::istreambuf_iterator` to read from start to finish in one go. `std::istreambuf_iterator` is an iterator that reads raw data from the file stream and puts it directly into a string. This way, the whole file is loaded fast without multiple reads or loops.

This method is quick because it handles the file as a single block. It’s simple and avoids the overhead of reading line by line. Alternatives include reading with `std::getline()` or using a `std::vector<char>` and reading chunks into it. But `std::istreambuf_iterator` is often faster when you need the whole file at once.

```cpp
std::future<std::string> result = std::async(std::launch::async, readFileAsync, argv[1]);
```

`std::async` starts `readFileAsync` in a new thread. The flag `std::launch::async` forces the function to run right away, not waiting until you call `result.get()`. This keeps the main thread free to keep working.

```cpp
std::cout << "File is being read asynchronously...\n";
```

While the file is being read, the main thread keeps going. This lets your program handle other tasks, like updating the UI, processing data, or waiting for user input. You’re not stuck waiting for the file read to finish.

```cpp
std::string content = result.get();
```

To get the file content, use `result.get()`. This line waits only if the file reading isn’t finished yet. If it’s done, it gives you the content right away. This way, the main thread pauses only when it really needs the result. This technique is important in cases where the program needs to handle tasks beyond reading input data, like updating interfaces, logging, or managing network connections. In competitive programming, these situations are rare since most programs focus on processing input data directly. However, for real-world applications where multitasking is crucial, asynchronous I/O keeps the program responsive while still handling file operations in the background.

Asynchronous I/O with `std::async` and `std::future` has clear advantages. Your program keeps running while files are being read or written. The main thread doesn’t wait. It keeps working. This reduces idle time and makes your program responsive.

It also lets your program handle many tasks at once. Reading and writing happen in the background. The main thread can do something else. This uses the full power of multi-core processors.

`std::async` and `std::future` are also simple. They fit right into the C++ Standard Library. No need for complex threading or low-level code. You can turn a blocking task into a background job with just a few lines. Your program stays efficient. Your code stays clean.

Asynchronous I/O boosts performance but doesn’t match `mmap` for direct memory access. It still uses standard file operations. The data has to be read into memory the usual way. This makes it slower than `mmap` for random access in large files or when you need the file to act like an array in memory.

**Unlike `mmap`, asynchronous I/O works across all major platforms: Windows, Linux, and macOS. It’s great for cross-platform development.** Real-time systems also benefit. They need quick responses and can’t afford to wait on I/O. In multitasking environments, like servers or data processing programs, asynchronous I/O shines. It handles multiple operations at once, boosting performance and keeping everything running smoothly.

Use this technique with care. Adding threads and async operations makes code complex. You need to sync threads right to avoid race conditions and data errors. That’s why we avoid this in competitive programming. Threads can speed things up, but they also have a cost. Sometimes the gain isn’t worth the hassle. Most competitive programming uses simple, straight I/O. In these cases, async I/O isn’t needed. It’s best for heavy I/O loads or when you can split reading and processing.

Parallel I/O fits when there are many read/write tasks or when the program needs to handle big data while still reading or writing files. You see this in AI competitions and hackathons. It helps when you work with large datasets or need fast input/output handling, like in "big data" challenges. But due to its complexity, save `std::async` and threading for when parallelism gives a clear edge over regular I/O. Keep it for the moments when it really counts.

### 2.4.4. Summary of Efficient Techniques for File I/O

| Function/Operation               | Most Efficient Technique                                                                    | Description                                                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reading from file (command line) | `mmap` for large files, `std::ifstream` or `fread` for small files                          | `mmap` maps files directly into memory, reducing I/O overhead for large files. `std::ifstream` and `fread` are better suited for smaller reads.          |
| Reading from standard input      | Disable synchronization with `std::ios::sync_with_stdio(false)` and `std::cin.tie(nullptr)` | Disables synchronization between C and C++ streams, speeding up `std::cin` operations by preventing unnecessary flushes of `std::cout`.                  |
| Writing to terminal              | `putchar` or `printf`                                                                       | `putchar` is optimal for single character output, while `printf` is often faster than `std::cout` for formatted output in competitive programming.       |
| Working with arrays              | `std::vector` with `std::span` (C++20)                                                      | `std::span` provides safe, efficient access to arrays and vectors without copies, offering bounds safety without performance penalties.                  |
| Data processing                  | `std::ranges` (C++20)                                                                       | `std::ranges` allows for lazy evaluation of data operations like filtering and transforming, avoiding unnecessary copies and improving efficiency.       |
| Parallel I/O                     | `std::async` with asynchronous read and write operations                                    | `std::async` allows reading and writing in parallel, improving performance in high I/O scenarios by leveraging multiple cores.                           |
| Vector manipulation              | `std::vector` with `std::transform` or `std::sort`                                          | Using C++ algorithms like `std::transform` and `std::sort` optimizes common operations without manual loops, taking advantage of compiler optimizations. |
| Handling large data volumes      | Manual buffering with `fread` and `fwrite`                                                  | `fread` and `fwrite` are highly efficient for handling large blocks of data, minimizing the number of I/O operations and system call overhead.           |

Efficient file handling is crucial, especially in competitive programming where input sizes can be large. Optimizing file I/O can be the difference between a solution that completes on time and one that fails. Using the right technique for each scenario ensures that your program handles data quickly and efficiently.

## 2.5. Fast Command-Line I/O in Competitive Programming

In competitive programming, inputs often come from the command line. First, you get the size of the array, then the array elements, separated by spaces. You must read this data fast and print the results quickly, especially with large datasets. Here's the fastest way to handle input and output on both Windows and Linux.

`scanf` and `printf` are thread-safe and handle complex output formatting with ease. Use them when you need reliable, clear I/O, especially in competitive programming or when formatting matters. They work well across platforms and are safe in multi-threaded code.

```cpp
#include <iostream>  // Include the standard input-output stream library
#include <vector>    // Include the vector library for dynamic arrays
#include <cstdio>    // Include C standard input-output header (optional in this context)

// Main function
int main() {
    // Disable synchronization between C and C++ standard streams for faster I/O operations
    std::ios::sync_with_stdio(false);
    // Untie cin from cout, ensuring cin is not flushed every time cout is used (improves speed)
    std::cin.tie(nullptr);

    // Declare an integer to store the size of the array
    int n;
    // Read the size of the array from standard input
    std::cin >> n;

    // Create a vector of integers with size 'n' to store the array elements
    std::vector<int> arr(n);

    // Loop to read 'n' integers into the vector
    for (int i = 0; i < n; ++i) {
        std::cin >> arr[i];  // Read each element and store it in the vector
    }

    // Loop to output each element of the vector
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";  // Print each element followed by a space
    }

    // Print a newline at the end of the output
    std::cout << std::endl;

    // Return 0 to indicate successful execution
    return 0;
}
```

We've seen this before, but let's review. Disabling I/O synchronization with `std::ios::sync_with_stdio(false);` speeds up the program. It stops `std::cin` and `std::cout` from syncing with `scanf` and `printf`. This saves time during input and output.

Unlinking `cin` and `cout` is another step. The command `std::cin.tie(nullptr);` stops `std::cout` from flushing before every input. Now, the program flushes only when you decide, not after every read. This gives you control and makes I/O faster.

**On both Windows and Linux, this code works well. But on Linux, it matters more.** Linux systems rely on I/O synchronization more, so turning it off boosts speed. On Windows, the effect is smaller but still helpful.

While `std::cin` and `std::cout` are fast after disabling sync, some Unix-based systems like ICPC allow even quicker methods. Using `scanf` and `printf`, which come from C, can boost speed further. But they come with risks. These functions don’t check the size of input properly, which can lead to buffer overflows and security vulnerabilities. If misused, they can let in dangerous data and cause crashes. This makes them powerful but risky tools in the wrong hands. Here's an alternative using `scanf` and `printf` for those who need the speed and are aware of the risks.

**Never, ever, under any circumstances, for any reason, use scanf or printf in professional code. Remember, we are only considering them for competitions.** Now that you know the risks, let’s look at an example.

```cpp
#include <cstdio>      // Include the C standard input-output header for scanf and printf functions
#include <vector>      // Include the vector library for using dynamic arrays

// Main function
int main() {
    // Declare an integer to store the size of the array
    int n;
    // Read the size of the array from standard input using scanf, a C function that reads formatted input
    scanf("%d", &n);

    // Create a vector of integers with size 'n' to store the array elements
    std::vector<int> arr(n);

    // Loop to read 'n' integers into the vector
    for (int i = 0; i < n; ++i) {
        // Read each element and store it in the vector using scanf
        scanf("%d", &arr[i]);
    }

    // Loop to output each element of the vector
    for (int i = 0; i < n; ++i) {
        // Print each element followed by a space using printf, which outputs formatted data
        printf("%d ", arr[i]);
    }

    // Print a newline at the end of the output to move the cursor to the next line
    printf("\n");

    // Return 0 to indicate that the program has executed successfully
    return 0;
}
```

> Using `scanf` and `printf` requires a clear understanding of how they handle data input and output. Both functions are part of the C standard library and are known for their speed but also for their lack of safety features compared to C++ alternatives. Below is a detailed explanation of how to use `scanf` and `printf`, with examples demonstrating reading arrays, characters, and printing integers, floats, strings, and characters.
>
> `scanf` reads formatted input from the standard input (keyboard). It requires format specifiers to interpret the data type of the input and pointers (memory addresses) to store the read values.
>
> To read an integer, use the `%d` specifier. You must provide the address of the variable using the `&` operator.
>
> ```cpp
> int n;
> scanf("%d", &n);  // Reads an integer from input and stores it in 'n'
> ```
>
> To read multiple integers into an array, `scanf` is used inside a loop. Here, `%d reads each integer, and `&arr[i]` passes the address of each element in the array.
>
> ```cpp
> std::vector<int> arr(n); // Creates a vector to store 'n' integers
>
> for (int i = 0; i < n; ++i) {
>    scanf("%d", &arr[i]);  // Reads each integer and stores it in the array
> }
> ```
>
> Use `%c` to read a single character. When reading characters, be cautious of whitespace issues; `scanf` may read unintended newline characters.
>
> ```cpp
> char ch;
> scanf(" %c", &ch);  // Reads a single character, the space before %c ignores any whitespace
> ```
>
> Strings are read using `%s`. scanf stops reading at the first whitespace, so it cannot handle multi-word strings without additional handling.
>
> ```cpp
> char str[100];
> scanf("%s", str);  // Reads a string and stores it in the character array 'str'
> ```
>
> For floats, use `%f`, and for doubles, use `%lf`. You still need to provide the variable's address.
>
> ```cpp
> float f;
> scanf("%f", &f);  // Reads a float value and stores it in 'f'
> ```
>
> printf outputs formatted data to the standard output (console). It also uses format specifiers to determine the data type of what is being printed.
>
> To print integers, use `%d`.
>
> ```cpp
> int x = 10;
> printf("%d\n", x);  // Prints the integer value of 'x'
> ```
>
> Printing arrays involves looping through each element and printing them one by one.
>
> ```cpp
> for (int i = 0; i < n; ++i) {
>    printf("%d ", arr[i]);  // Prints each element of the array followed by a space
> }
> printf("\n");  // Prints a newline after the array elements
> ```
>
> Use `%c` to print single characters.
>
> ```cpp
> char ch = 'A';
> printf("%c\n", ch);  // Prints the character 'A'
> ```
>
> To print strings, use `%s`.
>
> ```cpp
> char str[] = "Hello";
> printf("%s\n", str);  // Prints the string "Hello"
> ```
>
> Use `%f` for floats and `%lf` for doubles. You can also specify the precision.
>
> ```cpp
> float f = 3.14;
> printf("%.2f\n", f);  // Prints the float value with two decimal places
> ```
>
> When using `scanf`, variables are passed by reference using pointers, meaning you provide the memory address of the variable (`&` operator). This approach allows `scanf` to modify the variable directly. For example, `scanf("%d", &n);` reads an integer and stores it at the address of `n`.
>
> When printing with `printf`, you directly provide the value to print, without needing the address.

When thread safety isn’t a concern, faster options are available. Use `putchar` for single characters, and `puts` or `fputs` for unformatted strings. If you need formatted output without thread safety, go for `printf_unlocked`. For input, `getchar_unlocked` reads characters fast, skipping safety checks, ideal for large data or custom input parsing. It’s faster than `getchar`, which is thread-safe and still beats `scanf` for simple reads. Custom input with buffers using `fread` or `read` allows reading large blocks at once but requires manual data handling. Finally, `scanf_unlocked` is a faster, unsafe version of `scanf`, best when speed is critical, and threads don’t matter.

Let’s see an example with `printf_unlocked` and `scanf_unlocked`, swapping them in for `printf` and `scanf`. They’re faster because they skip safety checks. These functions work on Linux because they’re part of glibc, the library for Unix systems. You won’t find them on Windows. Windows libraries focus on safety and compatibility, not speed.

```cpp
#include <cstdio>      // Include the C standard input-output header for scanf_unlocked and printf_unlocked functions
#include <vector>      // Include the vector library for using dynamic arrays

// Main function
int main() {
    // Declare an integer to store the size of the array
    int n;
    // Read the size of the array from standard input using scanf_unlocked, a faster version of scanf without thread safety
    scanf_unlocked("%d", &n);

    // Create a vector of integers with size 'n' to store the array elements
    std::vector<int> arr(n);

    // Loop to read 'n' integers into the vector
    for (int i = 0; i < n; ++i) {
        // Read each element and store it in the vector using scanf_unlocked
        scanf_unlocked("%d", &arr[i]);
    }

    // Loop to output each element of the vector
    for (int i = 0; i < n; ++i) {
        // Print each element followed by a space using printf_unlocked, a faster variant without thread safety
        printf_unlocked("%d ", arr[i]);
    }

    // Print a newline at the end of the output to move the cursor to the next line
    printf_unlocked("\n");

    // Return 0 to indicate that the program has executed successfully
    return 0;
}
```

`printf_unlocked` and `scanf_unlocked` use the same arguments as their safer counterparts, `printf` and `scanf`. For `printf_unlocked`, you provide a format string followed by the values to print, just like `printf`. For `scanf_unlocked`, you give a format string and pointers to where the input should be stored. The key difference is speed and safety: `unlocked` versions skip thread checks, making them faster but unsafe in multi-threaded contexts. Use them when you control the environment and need every bit of speed.

`printf_unlocked`, `scanf_unlocked`, and `fgets` work only on Linux because they are part of glibc, the GNU C Library. They skip safety checks to be fast, which is why Windows doesn’t include them. Windows libraries stick to safer, thread-safe options. But you can still use these functions on Windows with compilers like Cygwin or MinGW. They bring Unix-like tools, including glibc, to Windows, letting you use these faster functions.

`printf_unlocked` and `scanf_unlocked` are useful in competitive programming on Linux, especially when you need complex output and don’t use threads. They are faster than `printf` and `scanf`, making them a good choice when speed matters. But they aren’t the fastest options. For even quicker I/O, use `putchar`, `puts`, `fputs`, `fgets`, and `getchar_unlocked`. These functions skip formatting and safety checks, giving you maximum speed when every millisecond counts.

`putchar` is the fastest for printing a single character. It skips formatting and works well for repetitive character output. `puts` is quick for printing strings; it doesn’t format and adds a newline at the end. `fputs` is similar to `puts` but doesn’t add a newline, giving you more control over output flow and can be faster in some cases. `fgets` reads strings efficiently, ideal for reading lines of text without worrying about formatting.

`getchar_unlocked` reads characters at high speed. It skips safety checks, making it faster than `getchar`. It’s best for reading large data sets or custom input parsing of integers and strings. But remember, it’s not safe in multi-threaded environments.

`getchar` reads one character at a time, faster than `scanf` for simple inputs or when paired with custom input buffers. It’s thread-safe but still quicker than more complex functions. These options give you raw speed when thread safety isn’t a priority.

Remember, these functions—`putchar`, `puts`, `fputs`, `fgets`, `getchar`, `getchar_unlocked`, `scanf_unlocked`, and `printf_unlocked`—are from C, not C++. They focus on raw speed and skip the safety features found in C++. Most of them, especially the unlocked versions, only work on Linux because they’re part of glibc. You won’t find them in standard Windows setups. Here's the code explained, showing the differences between reading and printing characters and numbers. This will help you understand the speed and use cases of each function.

```cpp
#include <cstdio>  // Include C standard I/O functions

int main() {
    // Use getchar_unlocked to read characters fast
    char ch = getchar_unlocked();  // Reads a single character quickly without thread safety checks, faster than getchar
    putchar(ch);                   // Prints the character without formatting, the fastest way to print characters

    // Read a string safely with fgets, unlike gets, which is unsafe
    char str[100];
    fgets(str, 100, stdin);        // Reads up to 99 characters including spaces, prevents overflow by limiting input size
    puts(str);                     // Prints the string with a newline, faster than printf for simple text output

    // Print formatted output quickly using printf_unlocked
    int n = 42;
    printf_unlocked("Number: %d\n", n); // Prints the integer with formatting, skips thread safety checks unlike printf

    // Use scanf_unlocked for fast input of integers
    int num;
    scanf_unlocked("%d", &num);    // Reads an integer quickly, faster than scanf because it avoids thread safety checks
    printf_unlocked("Read: %d\n", num); // Prints the integer using the fast, unlocked version of printf

    return 0;
}
```

`getchar_unlocked` takes no arguments except the call itself. It reads the next character from input, ignoring the checks that slow down `getchar`. You won’t be asked for anything fancy, just the raw input. It’s the tool for fast, unformatted character reads when time is tight.

`putchar` is its printing cousin. One argument: the character. It spits it out without asking questions. No formatting, no newline unless you add one yourself. Perfect when you need repeated character output with zero overhead.

`fgets` reads strings, but unlike `gets`, it won’t let you shoot yourself in the foot. It takes three arguments: the buffer, the size, and the input stream. The buffer catches the string, the size stops overflow, and stdin feeds it. It reads until it hits the size limit or a newline. That’s why it’s safer, faster, and reliable.

> `fgets` is your go-to when you need to read strings safely. It keeps things under control by asking for a buffer, the size of the buffer, and the input stream. It’s safer than `gets`, but you need to set it up right. Let’s break it down with examples for different inputs: arrays, strings, and matrices.
>
> `fgets` reads one line at a time, including spaces and stops at the newline or when it fills the buffer. It’s great for reading a line of text without overflow.
>
> ```cpp
> char str[100]; // Buffer to hold the input
> fgets(str, 100, stdin); // Reads up to 99 characters plus the null terminator
> ```
>
> It reads into `str`, stopping at $99$ characters or a newline. It adds a null terminator, so your string is always safe to use.
>
> When you need multiple strings, set up an array of buffers. Each `fgets` call reads into the next buffer.
>
> ```cpp
> char words[5][20]; // Array of 5 strings, each up to 19 characters plus null terminator
> for (int i = 0; i < 5; i++) {
>     fgets(words[i], 20, stdin); // Reads each line into the array
> }
> ```
>
> Here, each row of `words` holds one string. You read each line separately, and the 20-character limit keeps it safe.
>
> `fgets` can also help when reading a grid or matrix of characters, treating each row as a string.
>
> ```cpp
> char matrix[3][4]; // 3 rows, 4 columns
> for (int i = 0; i < 3; i++) {
>     fgets(matrix[i], 5, stdin); // Reads 4 characters and a newline
> }
> ```
>
> Each `fgets` call reads a row. The buffer size of $5$ includes 4 characters plus the null terminator.
>
> `fgets` doesn’t read numbers directly, but you can use it to grab the line and then convert the text.
>
> ```cpp
> char input[10];
> fgets(input, 10, stdin); // Read the input as a string
> int number = atoi(input); // Convert to integer using atoi
> ```
>
> This lets you handle the input safely as text before converting, reducing the risk of overflow.
>
> `fgets` puts control in your hands. It reads just enough to fill the buffer without spilling over. Use it for strings, arrays, and grids when you need precise control. It’s simple, fast, and doesn’t ask questions—it just gets the job done.

`puts` is simple: one string and it’s printed with a newline. No need to worry about formatting strings or handling variable types. It’s fast because it’s straightforward, great for dumping results to the screen when you don’t need the extra fluff.

`printf_unlocked` handles formatted output like its safer cousin, `printf`. You pass the format string and the values. Each %d, %c, or %f you throw at it gets processed and printed. But unlike `printf`, it skips thread safety, making it quicker in single-thread scenarios where control is yours.

`scanf_unlocked` is the input counterpart. You pass the format string and pointers to where the data goes. Use %d for integers, %c for characters, %s for strings—just like `scanf`. It reads fast but trusts you to handle safety. It won’t pause to double-check.

Each function has its role in competitive programming. Use `getchar_unlocked` when you need every character, no questions asked. Use `putchar` when you need to print those characters right back. `fgets` and `puts` give you quick ways to handle strings without the mess of formatting. And when you need more, `printf_unlocked` and `scanf_unlocked` step in, speeding through input and output without the safety nets. They’re fast because they trust you to manage the risks.

In competitive programming, every millisecond matters. Below is a table showing the fastest input and output functions, their uses, and where they work. Use it to pick the right tool for the job, depending on your platform and need for speed.

| Function             | Speed Order | Use Case                                                                                    | Platform      |
| -------------------- | ----------- | ------------------------------------------------------------------------------------------- | ------------- |
| **Input Functions**  |             |
| `getchar_unlocked`   | 1           | Fastest for reading characters without safety checks; ideal for fast unformatted input.     | Linux         |
| `fgets`              | 2           | Reads strings safely with control over buffer size; good for line input.                    | Linux/Windows |
| `scanf_unlocked`     | 3           | Fast input of formatted data without thread safety; ideal when speed is critical.           | Linux         |
| `getchar`            | 4           | Thread-safe reading of single characters; slower but safer than getchar_unlocked.           | Linux/Windows |
| `scanf`              | 5           | Standard formatted input; safer but slower due to safety checks.                            | Linux/Windows |
| `std::cin`           | 6           | Standard input in C++, slower than C functions but safer and easier to use.                 | Linux/Windows |
|                      |             |
| **Output Functions** |             |
| `putchar`            | 1           | Fastest for printing single characters without formatting.                                  | Linux/Windows |
| `puts`               | 2           | Quick for printing strings with a newline; faster than printf for simple strings.           | Linux/Windows |
| `fputs`              | 3           | Similar to puts but without the newline, giving more control over output flow.              | Linux/Windows |
| `printf_unlocked`    | 4           | Formatted output without thread safety; faster in controlled, single-threaded environments. | Linux         |
| `printf`             | 5           | Standard formatted output; slower due to thread safety and formatting checks.               | Linux/Windows |
| `std::cout`          | 6           | Standard output in C++, slower than C functions; handles complex formatting easily.         | Linux/Windows |

## 2.6. Boosting I/O Efficiency with fread and fwrite for Bulk Data

`fread` and `fwrite` are made for speed with big data. They handle data in blocks, reducing the system calls that drag performance down. Large inputs are common in competitions focused on big data, AI, and statistics. That’s why we study these techniques. This guide is about competitive programming, but we’re also shaping good professionals. Use `fread` and `fwrite` when you need to move big chunks of data fast, and keep in mind the skills you build here go beyond the contest.

`fread` reads bytes from a file or `stdin` into a buffer. It grabs everything in one go, reducing overhead. This is ideal when you need to pull in large chunks of data fast.

```cpp
#include <cstdio>      // Include the C standard input-output library for functions like fread and putchar

// Main function
int main() {
    // Declare a buffer array of 1024 characters to temporarily hold the data read from input
    char buffer[1024];  // 1 KB manual buffer

    // Use fread to read data from standard input (stdin) into the buffer
    // fread returns the total number of elements successfully read, stored in bytesRead
    size_t bytesRead = fread(buffer, 1, sizeof(buffer), stdin);

    // Loop through each byte that was read into the buffer
    for (size_t i = 0; i < bytesRead; ++i) {
        // Use putchar to output each character from the buffer to standard output (stdout)
        // putchar prints one character at a time from the buffer to the console
        putchar(buffer[i]);
    }

    // Return 0 to indicate successful execution of the program
    return 0;
}
```

`fread` reads up to the specified number of items and stores them in your buffer. In the example, `fread(buffer, 1, sizeof(buffer), stdin)` reads up to $1024$ bytes and stores them in `buffer`. It returns the number of bytes read, so you know exactly what was processed.

Let’s break down the line `size_t bytesRead = fread(buffer, 1, sizeof(buffer), stdin);` step by step.

Starting from `size_t`: This is an unsigned integer type. It’s used to represent sizes and counts without negative values. `size_t` is used because it’s safe for indexing and size calculations. It automatically matches the correct size for the system, 32-bit or 64-bit. We use `size_t` here because `fread` returns the number of items read, and it’s always a positive number.

Following we have `fread`: This function reads data from a stream into a buffer. It’s designed for bulk reading, pulling data in chunks instead of piece by piece.

**Arguments of `fread`**:

1. **`buffer`**: This is the destination where `fread` will store the data it reads. It’s an array of characters, `char buffer[1024]`. This buffer holds up to $1024$ bytes of input data. Think of it as a container where `fread` dumps what it reads.

2. **`1`**: This is the size of each element to read. Here, it’s set to `1`, which means one byte at a time. It’s a straightforward way to read raw bytes without worrying about formatting or specific data types.

3. **`sizeof(buffer)`**: This tells `fread` how many bytes in total to read. `sizeof(buffer)` calculates the size of the buffer, which is $1024$ bytes. So, `fread` will attempt to read up to $1024$ bytes in one go. It won’t read more than the buffer can hold, keeping everything safe.

4. **`stdin`**: This is the input stream—standard input, usually the keyboard or data redirected from a file. `fread` pulls data from this stream and fills the buffer.

Putting it all together, `fread(buffer, 1, sizeof(buffer), stdin)` reads up to $1024$ bytes of data from standard input and stores it in the buffer. It does this in one operation, reducing the time spent on input compared to reading character by character. **The buffer is stored in dynamic memory, meaning it lives in the heap and grows or shrinks as needed, giving you control over large data.**

For fast writing, `fwrite` is your go-to. It pushes data from your buffer to a file or `stdout` in one sweep. It’s fast because it doesn’t stop for formatting.

```cpp
#include <cstdio>      // Include the C standard input-output library for functions like fwrite
#include <vector>      // Include the vector library (not used in this code but commonly included for dynamic arrays)

// Main function
int main() {
    // Define a constant character pointer pointing to a string of data
    const char* data = "Outputting large blocks of data quickly\n";
    // Calculate the size of the data string using strlen
    // strlen returns the length of the string excluding the null terminator
    size_t dataSize = strlen(data);

    // Write the data from the buffer to standard output using fwrite
    fwrite(data, 1, dataSize, stdout);

    // Return 0 to indicate successful execution of the program
    return 0;
}

```

`fwrite` sends data from your buffer in a single operation. `fwrite(data, 1, dataSize, stdout)` writes `dataSize` bytes to `stdout`, avoiding the slowdown of smaller writes. Use `fread` and `fwrite` when you need raw speed without the fluff of formatting. They’re built for bulk I/O and shine when performance is key. Let's understand this code:

Starting from `const char* data`. It’s a pointer to a constant string. The string lives in read-only memory, which means you can’t change it. This string holds the message you want to send out to the output. It’s fixed, and it’s fast.

Following we have `size_t dataSize = strlen(data);`: `strlen` counts the characters in the string, skipping the `null` terminator. It tells you exactly how many bytes `fwrite` will handle. We use `size_t` because it’s safe on any system, whether you’re on a 32-bit or 64-bit machine. It handles sizes without fuss.

Finally, we get `fwrite(data, 1, dataSize, stdout);`: `fwrite` pushes the data straight from the buffer to the output. It’s designed to move data in bulk, making it faster and more efficient. It's arguments are:

1. **`data`**: The buffer with your data. It’s the source.
2. **`1`**: The size of each piece of data to write, in bytes. Here, it’s one byte per character.
3. **`dataSize`**: The number of bytes `fwrite` needs to write, which `strlen` calculated for us.
4. **`stdout`**: The output stream, usually your screen or console.

This setup makes `fwrite` do the work in a single shot. It doesn’t break the data into tiny pieces. It just moves it all at once, cutting down on system calls that waste time.

`fwrite` handles data in blocks, unlike `putchar` or `printf` that deal with one character at a time. This bulk operation makes it quicker and perfect for competitive programming when you need to dump large amounts of data fast. It’s built to speed up the process, especially when every second counts.

## 2.7. The Last Trick: Namespaces

In C++, **namespaces** help organize your code and keep names from clashing. They are essential in big projects or when using multiple libraries that might have functions, classes, or variables with the same name. A namespace sets up a scope, letting you define functions, classes, and variables without fear of conflicts.

A **namespace** is a space where identifiers (names of types, functions, variables, etc.) live. It allows you to use the same names in different parts of your program or across libraries without mixing things up.

```cpp
namespace MyNamespace {
    void myFunction() {
        // Implementation
    }

    class MyClass {
    public:
        void method();
    };
}
```

In this example, `MyNamespace` wraps `myFunction` and `MyClass`, keeping them separate from names in other namespaces. This prevents collisions and keeps your code clean and clear.

To access elements inside a namespace, use the **scope resolution operator** `::`.

The _scope resolution operator_ (`::`) in C++ tells the compiler exactly where to find an element. It cuts through the noise when names are the same but live in different places. Whether it’s a function, variable, or class, `::` points to the right spot. If a function is inside a namespace, `::` pulls it out from the right one. Inside a class, it defines functions outside the declaration or accesses static members.

In competitive programming, you see `::` all the time with `std::cout` or `std::vector`. It makes sure you’re using the standard library, keeping things clear and avoiding mix-ups with your own code or other libraries. Even though it’s not always needed in small, quick code, in bigger projects, `::` is crucial. It keeps references straight, especially when names overlap in different parts of the program.

```cpp
int main() {
    // Calling the function inside MyNamespace
    MyNamespace::myFunction();

    // Creating an object of the class inside the namespace
    MyNamespace::MyClass obj;
    obj.method();

    return 0;
}
```

### 2.7.1 `using namespace std;`

The **std** namespace is the core of the C++ Standard Library. It holds everything you use daily, like `std::vector`, `std::cout`, `std::string`, and much more.

When you write `using namespace std;`, you’re telling the compiler to pull everything from `std` into your code. It saves you from typing `std::` every time. This makes the code shorter and easier to read, especially in small programs or quick examples. It’s a important time-saver when you’re in a rush, like during competitions.

Let's see a simple example without `using namespace std;`:

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    for (const int& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

Now compare the earlier code with an example with `using namespace std;`\*\*:

```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> numbers = {1, 2, 3, 4, 5};
    for (const int& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    return 0;
}
```

The first version, with `using namespace std;`, has $24$ caracteres more than the version without `using namespace std;`. This in only $10$ lines. Think about a code complex enough to build a graph and search it for a value.

`using namespace std;` makes your code shorter and easier to read, but it has its downsides. In bigger projects or when using multiple libraries, it raises the risk of name conflicts. Different namespaces might have elements with the same name, and that leads to confusion. It can make your code harder to maintain and understand because you lose track of where things come from. That’s why it’s best to avoid `using namespace std;` in production code, especially in large or shared projects.

To avoid the risks of `using namespace std;`, import only what you need from the `std` namespace. Instead of pulling in everything, bring in just the functions and types you use, like `std::cout` and `std::vector`. This keeps your code clear and reduces the chance of name conflicts, while still making it concise.

```cpp
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

int main() {
    vector<int> numbers = {1, 2, 3, 4, 5};
    for (const int& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    return 0;
}
```

Keep your code clean and easy to manage. Don’t use `using namespace std;` in header files. It forces every file that includes the header to pull in the `std` namespace, raising the risk of conflicts. If you must use `using`, limit it to a small scope, like inside a function. This keeps the impact small. Stay consistent with how you handle namespaces in your project. It makes the code easier to read and work with. In competitive programming, headers are rare, but in bigger projects, this discipline keeps things in order.

**In competitive programming, you don’t need custom namespaces beyond `std`. The code is small, fast, and used once. Custom namespaces add complexity with no real benefit. They’re made to avoid conflicts in big projects with many libraries, but in competitions, you rarely have that problem. The focus is on speed and simplicity, not on managing names. The same goes for object-oriented programming, it adds overhead that slows you down. Stick to the basics. Leave namespaces and OOP for large projects where they make sense. In competitions, keep it simple and keep it fast.**
