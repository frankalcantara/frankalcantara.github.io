---
author: Frank
beforetoc: '[Anterior](2024-09-20-27-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-29-Sem-T%C3%ADtulo.md)'
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
title: Process the adjustments
toc: true
---
# Process the adjustments
for _ in range(q):
    adjustment, index = map(int, input().strip('[]').split(','))

    # Check if the current value at the index is even before the adjustment
    if inventory[index] % 2 == 0:
        even_sum -= inventory[index]  # Remove from the sum if it was even

    # Apply the adjustment
    inventory[index] += adjustment

    # Check if the new value at the index is even after the adjustment
    if inventory[index] % 2 == 0:
        even_sum += inventory[index]  # Add to the sum if it is now even

    # Print the updated sum of even numbers
    print(even_sum)
```

In this pseudocode, we observe the following steps:

1. **Input Reading**: First, we read the value of $n$ (inventory size) and the integer array $inventory$. Then, we read the number of adjustments $q$ and each adjustment.
2. **Initial Calculation**: We calculate the initial sum of even numbers in the $inventory$ array.
3. **Processing Adjustments**: For each adjustment, we check if the value at the affected index is even before the adjustment. If it is, we remove that value from the sum of even numbers. We then apply the adjustment, and if the new value at the index is even, we add it to the sum of even numbers.
4. **Output**: After each adjustment, we print the updated sum of even numbers.

For this solution the time complexity is $O(n + q)$, where $n$ is the size of the $inventory$ array and $q$ is the number of adjustments. The code processes each adjustment in constant time since the sum is maintained incrementally.

```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Hardcoded input values
    int n = 6, q = 4;
    vector<int> inventory = { 10, 3, 5, 6, 8, 2 };
    vector<pair<int, int>> adjustments = { {3, 1}, {-4, 0}, {2, 3}, {-3, 4} };

    // Initial sum of even numbers
    int even_sum = 0;
    for (int i = 0; i < n; ++i)
        if (!(inventory[i] & 1)) even_sum += inventory[i]; // Check if even

    // Process adjustments
    for (int i = 0; i < q; ++i) {
        int adj = adjustments[i].first, idx = adjustments[i].second;
        if (!(inventory[idx] & 1)) even_sum -= inventory[idx]; // Subtract if even
        inventory[idx] += adj;  // Apply adjustment
        if (!(inventory[idx] & 1)) even_sum += inventory[idx]; // Add if now even
        cout << even_sum << '\n';
    }
}
```

> In C++, the expression `inventory[i] & 1` is a bitwise operation that checks whether the value at index $i$ in the `inventory` array is **even or odd**.

**Detailed Explanation**:

- **`&` Operator**: This is the **bitwise AND** operator. It performs an AND operation on the binary representation of two numbers, comparing corresponding bits.
- **`1` in Binary**: The number $1$ in binary is represented as $0000...0001$ (depending on the size of the integer). Since only the least significant bit is set to $1$, this operation focuses specifically on the least significant bit of `inventory[i]`.
- **Bitwise AND (`&`)**: The bitwise AND operator returns $1$ if both corresponding bits of the operands are $1$. For other cases, it returns $0$.

In the naïve code this operation checks whether the **least significant bit** (LSB) of the value at `inventory[i]` is $1$ or $0$, which directly indicates if the number is odd or even:

- If the result of `inventory[i] & 1` is $1$, the number is **odd**.
- If the result of `inventory[i] & 1` is $0$, the number is **even**.

This bitwise approach is **faster** than using the modulo operation (`inventory[i] % 2 == 0`) for determining even/odd status, as it avoids division and is optimized at the hardware level. Consider the following values for `inventory[i]`:

- For `inventory[i] = 6` (binary: $110$), the operation:
  - $6 \& 1 = 0$
  - Since the result is $0$, 6 is **even**.
- For `inventory[i] = 5` (binary: $101$), the operation:

  - $5 \& 1 = 1$
  - Since the result is $1$, 5 is **odd**.

> ### Bitwise Operations in C++
>
> Bitwise operations in C++ manipulate individual bits of integers. These operations are low-level but powerful, allowing programmers to perform tasks like toggling, setting, or clearing specific bits. They are commonly used in scenarios where performance is critical, such as embedded systems, cryptography, and competitive programming.
>
> C++ provides several bitwise operators that work directly on the binary representation of numbers. These operators include:
>
> - **AND (`&`)**
> - **OR (`|`)**
> - **XOR (`^`)**
> - **NOT (`~`)**
> - **Left Shift (`<<`)**
> - **Right Shift (`>>`)**
>
> **Bitwise AND (`&`)**
>
> The **bitwise AND** operator compares each bit of its operands and returns $1$ if both bits are $1$. Otherwise, it returns $0$.
>
> ```cpp
> int a = 6;    // Binary: 110
> int b = 3;    // Binary: 011
> int result = a & b;  // result = 2 (Binary: 010)
> ```
>
> In this code we have: the binary representation of 6 is $110$ and the binary representation of 3 is $011$. When performing the AND operation:
>
> - $1 \& 0 = 0$
> - $1 \& 1 = 1$
> - $0 \& 1 = 0$
>
> Therefore, the result is $010$ in binary, which is $2$ in decimal.
>
> **Bitwise OR (`|`)**
>
> The **bitwise OR** operator compares each bit of its operands and returns $1$ if either of the bits is $1$.
>
> ```cpp
> int a = 6;    // Binary: 110
> int b = 3;    // Binary: 011
> int result = a | b;  // result = 7 (Binary: 111)
> ```
>
> The binary representation of 6 is $110$ and the binary representation of 3 is $011$. When performing the OR operation:
>
> - $1 \| 0 = 1$
> - $1 \| 1 = 1$
> - $0 \| 1 = 1$
>
> Therefore, the result is $111$ in binary, which is $7$ in decimal.
>
> **Bitwise XOR (`^`)**
>
> The **bitwise XOR** (exclusive OR) operator compares each bit of its operands and returns $1$ if the bits are different, and $0$ if they are the same.
>
> ```cpp
> int a = 6;    // Binary: 110
> int b = 3;    // Binary: 011
> int result = a ^ b;  // result = 5 (Binary: 101)
> ```
>
> The binary representation of 6 is $110$ and the binary representation of 3 is $011$. When performing the XOR operation:
>
> - $1 \oplus 0 = 1$
> - $1 \oplus 1 = 0$
> - $0 \oplus 1 = 1$
> - Therefore, the result is $101$ in binary, which is $5$ in decimal.
>
> **Bitwise NOT (`~`)**
>
> The **bitwise NOT** operator inverts all the bits of its operand. It converts $1$s to $0$s and $0$s to $1$s.
>
> ```cpp
> int a = 6;    // Binary: 00000000 00000000 00000000 00000110 (32-bit system)
> int result = ~a;  // result = -7 (Binary: 11111111 11111111 11111111 11111001)
> ```
>
> The binary representation of 6 is $0000...0110$ (with 32 bits). The NOT operation flips each bit: $~110$ becomes $111...1001$.The result is the two's complement representation of $-7$.
>
> **Left Shift (`<<`)**
>
> The **left shift** operator shifts the bits of its first operand to the left by the number of positions specified by the second operand. This effectively multiplies the number by powers of 2.
>
> ```cpp
> int a = 3;    // Binary: 00000000 00000000 00000000 00000011
> int result = a << 1;  // result = 6 (Binary: 00000000 00000000 00000000 00000110)
> ```
>
> The binary representation of 3 is $011$. Shifting it left by $1$ position results in $110$, which is $6$ in decimal. Shifting by $n$ positions is equivalent to multiplying by $2^n$.
>
> **Right Shift (`>>`)**
>
> The **right shift** operator shifts the bits of its first operand to the right by the number of positions specified by the second operand. This effectively divides the number by powers of 2 (for positive integers).
>
> ```cpp
> int a = 6;    // Binary: 00000000 00000000 00000000 00000110
> int result = a >> 1;  // result = 3 (Binary: 00000000 00000000 00000000 00000011)
> ```
>
> The binary representation of 6 is $110$. Shifting it right by $1$ position results in $011$, which is $3$ in decimal.Shifting by $n$ positions is equivalent to dividing by $2^n$ (for non-negative integers).
>
> ### Summary Table of Bitwise Operations
>
> | Operation   | Symbol | Effect                                                                    |
> | ----------- | ------ | ------------------------------------------------------------------------- | -------------------------------------------------- |
> | Bitwise AND | `&`    | Compares bits, returns $1$ if both are $1$.                               |
> | Bitwise OR  | `      | `                                                                         | Compares bits, returns $1$ if at least one is $1$. |
> | Bitwise XOR | `^`    | Compares bits, returns $1$ if bits are different.                         |
> | Bitwise NOT | `~`    | Inverts each bit (turns $0$s into $1$s and $1$s into $0$s).               |
> | Left Shift  | `<<`   | Shifts bits to the left, multiplying by powers of 2.                      |
> | Right Shift | `>>`   | Shifts bits to the right, dividing by powers of 2 (for positive numbers). |
>
> **Applications of Bitwise Operations**:
>
> 1. **Efficiency**: Bitwise operations are faster than arithmetic operations, making them useful in performance-critical code.
> 2. **Bit Manipulation**: They are commonly used for tasks such as toggling, setting, and clearing bits in low-level programming, such as working with hardware or network protocols.
> 3. **Masking and Flagging**: Bitwise operators are often used to manipulate flags in bitmasks, where individual bits represent different conditions or options.

### 4 - Sales Target Analysis

You are tasked with analyzing sales data to determine how many subarrays of daily sales sum to a multiple of a target value $T$ . The sales data is recorded in an array sales , and you need to calculate how many contiguous subarrays of sales have a sum divisible by $T$ .

**Input Format**:

The first line contains two integers $n$ (the size of the sales array) and $T$ (the target value).
The second line contains $n$ integers, representing the daily sales data.
Constraints:

$$
1 \leq n \leq 10^5 \\
1 \leq T \leq 10^4 \\
-10^4 \leq \text{sales}[i] \leq 10^4
$$

Output Format:

Output a single integer representing the number of subarrays whose sum is divisible by $T$ .

Example **Input**: 6 5 4 5 0 -2 -3 1

Example **Output**: 7

Explanation:

There are $7$ subarrays whose sum is divisible by $T=5$ :

```text
[4,5,0,−2,−3,1]
[5]
[5,0]
[5,0,−2,−3]
[0]
[0,−2,−3]
[−2,−3]
```

**Input Method**:

The input is provided via command-line arguments.

#### Naïve Code

The algorithm works as follows:

1. We define a function `count_divisible_subarrays` that takes the `sales` array and target value `T` as inputs.

2. We use two nested loops to generate all possible subarrays:

   - The outer loop (`start`) determines the starting index of each subarray.
   - The inner loop (`end`) determines the ending index of each subarray.

3. For each subarray, we calculate the sum (`subarray_sum`) and check if it's divisible by `T`.

4. If a subarray sum is divisible by `T`, we increment our `count`.

5. After checking all subarrays, we return the total `count`.

6. Outside the function, we read the input from command-line arguments:

   - `n` and `T` are the first two arguments.
   - The `sales` array is constructed from the remaining arguments.

7. We call our function with the input data and print the result.

Here's a simple algorithm to solve the Sales Target Analysis problem, described in English with Python as pseudo-code:

1. Define a function to count divisible subarrays:

   ```python
   def count_divisible_subarrays(sales, T):
   count = 0
   for start in range(len(sales)):
       subarray_sum = 0
       for end in range(start, len(sales)):
           subarray_sum += sales[end]
           if subarray_sum % T == 0:
               count += 1
   return count
   ```

2. Read input from command-line arguments:

   ```python
   n, T = map(int, sys.argv[1:3])
   sales = list(map(int, sys.argv[3:]))
   ```

3. Call the function and print the result:

   ```python
   result = count_divisible_subarrays(sales, T)
   print(result)
   ```

The time complexity of this algorithm is $O(n^2)$, where $n$ is the number of elements in the sales array. This is because we're checking all possible subarrays, which requires two nested loops.

While this solution is straightforward and works for small inputs, it may not be efficient for large datasets (up to $10^5$ elements as per the problem constraints). An optimized solution using prefix sums and modular arithmetic could solve this problem in $O(n)$ time, but that's beyond the scope of a beginner's approach.

```cpp
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    // Read n and T from command-line arguments
    int n = stoi(argv[1]);
    int T = stoi(argv[2]);

    // Read the sales data from command-line arguments
    vector<int> sales(n);
    for (int i = 0; i < n; ++i) {
        sales[i] = stoi(argv[i + 3]);
    }

    // Function to count subarrays with sum divisible by T
    int count = 0;

    // Iterate over all possible subarrays
    for (int start = 0; start < n; ++start) {
        int subarray_sum = 0;
        for (int end = start; end < n; ++end) {
            subarray_sum += sales[end];
            if (subarray_sum % T == 0) {
                count++;
            }
        }
    }

    // Output the result
    cout << count << endl;

    return 0;
}
```

> Both `atoi` and `stoi` are functions used in C++ to convert strings to integers, but they differ in terms of capabilities, safety, and how they handle errors.
>
> **`atoi` (ASCII to Integer)**
>
> The `atoi` function is a legacy C-style function that converts a C-string (i.e., a character array) to an integer. It is part of the `<cstdlib>` header.
>
> ```cpp
> int atoi(const char* str);
> ```
>
> - **Parameter**: The function takes a single argument, `str`, which is a pointer to a null-terminated C-string (an array of characters).
> - **Return**: The function returns the integer representation of the string, or `0` if the conversion fails.
>
> ```cpp
> #include <iostream>
> #include <cstdlib>
>
> int main() {
>     const char* str = "12345";
>     int num = atoi(str);
>     std::cout << num << std::endl;  // **Output**: 12345
>     return 0;
> }
> ```
>
> - `atoi` reads the string and converts valid characters (i.e., digits) into an integer.
> - If the string contains any non-numeric characters, `atoi` stops reading at the first invalid character and returns the number formed so far.
> - If the string does not contain a valid integer at all, it returns `0`.
>
> **Limitations**:
>
> - **Error Handling**: `atoi` does not provide any error checking or exception handling. If the input is invalid (e.g., an empty string or a string with non-numeric characters), it simply returns `0`, making it hard to detect if an error occurred.
> - **Overflow/Underflow**: There is no way to detect if the resulting integer overflows or underflows the limits of the `int` type.
>
> **`stoi` (String to Integer)**:
>
> The `stoi` function is a C++-specific function that is more robust and safer than `atoi`. It is part of the `<string>` header and can convert a `std::string` or C-string to an integer. Unlike `atoi`, `stoi` provides error handling and works with the `std::string` class.
>
> ```cpp
> int stoi(const std::string& str, size_t* idx = 0, int base = 10);
> ```
>
> - **Parameters**:
>   - `str`: A reference to a `std::string` object or a C-string to be converted.
>   - `idx` (optional): A pointer to a `size_t` variable where the function stores the index of the first invalid character. If no invalid character is found, this is ignored.
>   - `base` (optional): The base of the number system (default is base 10). It can handle different bases like hexadecimal (base 16), octal (base 8), etc.
> - **Return**: The function returns the integer representation of the string.
> - **Exception**: Throws `std::invalid_argument` if no conversion can be performed, or `std::out_of_range` if the resulting integer is out of range for the `int` type.
>
> ```cpp
> #include <iostream>
> #include <string>
>
> int main() {
>     std::string str = "6789";
>     int num = stoi(str);
>     std::cout << num << std::endl;  // **Output**: 6789
>
>     // Handling conversion with an invalid character
>     try {
>         std::string invalid_str = "123abc";
>         int invalid_num = stoi(invalid_str);  // Throws an exception
>     } catch (const std::invalid_argument& e) {
>         std::cout << "Invalid argument: " << e.what() << std::endl;
>     }
>
>     return 0;
> }
> ```
>
> - `stoi` starts from the beginning of the string and converts as many valid numeric characters as it can into an integer.
> - If the string contains any non-numeric characters, the function will throw an exception (`std::invalid_argument`).
> - If the resulting integer exceeds the bounds of the `int` type, `std::out_of_range` is thrown.
>
> **Advantages**:
>
> - **Error Handling**: Unlike `atoi`, `stoi` provides proper error handling through exceptions, making it more robust.
> - **Base Conversion**: `stoi` can convert numbers in different bases (e.g., hexadecimal or octal) by specifying the `base` parameter.
> - **Range Checking**: It handles integer overflow/underflow and throws `std::out_of_range` if the value is too large or too small for the `int` type.
>
> **Example with Base Conversion**:
>
> ```cpp
> std::string hex_str = "1A";  // Hexadecimal string
> int num = stoi(hex_str, nullptr, 16);  // Converts from base 16 (hex) to base 10
> std::cout << num << std::endl;  // **Output**: 26
> ```
>
> **Summary of Differences**:
>
> | Feature            | `atoi`                        | `stoi`                                                |
> | ------------------ | ----------------------------- | ----------------------------------------------------- |
> | Input              | C-string (`const char*`)      | `std::string` or C-string                             |
> | Error Handling     | Returns `0` for invalid input | Throws `std::invalid_argument` or `std::out_of_range` |
> | Base Conversion    | Only base 10                  | Supports multiple bases                               |
> | Exception Safety   | No                            | Yes (uses C++ exceptions)                             |
> | Overflow/Underflow | No handling                   | Detects and throws `std::out_of_range`                |

