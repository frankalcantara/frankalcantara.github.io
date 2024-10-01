---
author: Frank
beforetoc: |-
    [Anterior](2024-09-24-7-7.-Working-with-Vector-and-Matrix.md)
    [Próximo](2024-09-24-9-9.-Time-and-Space-Complexity-in-Competitive-Programming.md)
categories:
    - Matemática
    - Linguagens Formais
    - Programação
description: Explore efficient data manipulation in C++ using Span and Ranges with practical examples, performance insights, and coding tips.
draft: null
featured: false
image: assets/images/prog_dynamic.jpeg
keywords:
    - Developer Tips
lastmod: 2024-09-30T01:33:22.853Z
layout: post
preview: In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Competitive Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and boost algorithm efficiency.
published: false
rating: 5
slug: efficient-data-manipulation-in-C++
tags:
    - Coding Examples
    - Algorithm Optimization
    - Practical Programming Guide
title: 7. Efficient Data Manipulation in C++
toc: true
---

# 3. Optimizing Loops and Data Structures

In competitive programming, mastering the basics of C++20 is essential for writing efficient and optimized code. This section focuses on fundamental control structures like loops and essential data structures such as vectors and matrices. Both elements play a important role in solving problems under time constraints.

C++20 introduces several features that enhance the flexibility and performance of loops. Techniques like range views and parallel execution allow programmers to process data with greater efficiency. Whether you are dealing with small arrays or large datasets, choosing the right loop can significantly impact the runtime of your solution.

Alongside loops, vectors and matrices serve as the foundation for storing and manipulating data. Understanding how to effectively use these data structures, combined with modern C++ features, allows you to handle complex computations with ease.

In the following sections, we will explore these elements in-depth, providing examples and performance considerations to help you develop competitive programming skills using C++20.

# 3.1. Working with Vector and Matrix

Vectors are flexible. They change size and are easy to use. You can insert, remove, and resize them with little effort. They work well for many tasks in competitive programming. Vectors hold one-dimensional data, but can also represent matrices. These two-dimensional vectors are often used for grids, tables, or simulations.

Matrices, built from vectors, handle multi-dimensional problems. They are good for game boards, adjacency matrices, and dynamic programming. You can change rows and columns, transpose the matrix, or access submatrices. Vectors and matrices give you control over how you store and process data. Starting with vectors.

In C++20, vectors have several important features that make them a powerful tool for managing collections of elements. First, vectors dynamically resize. This means they grow automatically when you add elements beyond their current capacity. You don’t need to manually manage memory like you would with arrays.

Vectors also provide random access. You can access any element by its index using `[]`, just as you would with a regular array. This makes it easy to work with elements directly without needing to traverse the vector.

## 3.1.1. Vetores, basic operations

In C++, the `std::vector` class is part of the Standard Template Library (STL). It is a dynamic array that manages collections of elements. Unlike arrays, vectors resize when elements are added or removed. This makes them useful in competitive programming when data sizes change during execution.

Vectors offer random access to elements. They support iteration with iterators and allow dynamic resizing. They provide functions like `push_back`, `pop_back`, `insert`, `erase`, and `resize`. These make managing data easier without manual memory handling.

The following example shows how to create a vector in C++20, reserve memory, and initialize elements. We start by creating a vector, reserving space for $10$ elements, and then resizing it to hold $10$ elements initialized to $0$.

```cpp
std::vector<int> vec;
vec.reserve(10); // Reserves memory for 10 elements without changing size
vec.resize(10, 0); // Resizes the vector to 10 elements, all initialized to 0
```

`vec.reserve(10);` reserves memory for $10$ elements but doesn’t affect the vector’s size. This means no elements are created yet, but space is allocated in advance to avoid future reallocations. Then, `vec.resize(10, 0);` creates $10$ elements, all initialized to $0$, using the reserved memory.

The class template `std::vector` is part of the Standard Template Library (STL) and is a dynamic array. It allows for efficient management of collections of elements. When you write `std::vector<int>`, you are creating a vector where each element is of type `int`.

`std::vector` is a _generic_ class. This means it can store elements of any type, not just integers. For example, `std::vector<double>` stores `double` values, and `std::vector<std::string>` stores strings. The class is defined with a template parameter that specifies the type of the elements.

Here’s an example of creating vectors with different types:

```cpp
std::vector<int> intVec; // Vector of integers
std::vector<double> doubleVec; // Vector of doubles
std::vector<std::string> stringVec; // Vector of strings
```

When you create a `std::vector<int>`, the compiler generates a specialized version of the `std::vector` class for integers. The same applies to any other type you use with `std::vector`. This is the power of _generic programming_, the ability to write code that works with any type while maintaining type safety.

One important feature of `std::vector` is efficient memory management. By using methods like `reserve` and `shrink_to_fit`, you can control the vector’s capacity. Reserving memory early ensures that the vector has enough space for future elements. This avoids multiple reallocations, which can be expensive because each reallocation involves copying the entire vector to a new memory location. When you know in advance how many elements the vector will need, calling `reserve` improves performance by minimizing these costly reallocations. We saw `reserve` before so let's see `shrink_to_fit`.

`shrink_to_fit` requests the vector to reduce its capacity to match its size. This helps free unused memory after adding or removing elements. It’s not guaranteed, but it allows the system to optimize memory usage.

Here’s a simple example:

```cpp
std::vector<int> vec;
vec.reserve(100);  // Reserve space for 100 elements
vec.resize(10);    // Resize to hold 10 elements
vec.shrink_to_fit(); // Reduce capacity to fit size
```

In this example, we reserve space for $100$ elements, resize it to $10$, and then call `shrink_to_fit` to match capacity to the actual number of elements. It ensures memory is not wasted on unused space.

Vectors classe also come with several built-in functions that make them easy to use. For example, `push_back` allows you to add an element to the end of the vector, while `pop_back` removes the last element. The `insert` function lets you add elements at specific positions within the vector.

For example, inserting a value into a vector' end:

```cpp
intVec.push_back(5); // Adds the value 5 to the end of the vector
```

`intVec.push_back(5);` appends the value $5$ to the end of the vector. This function resizes the vector if necessary, allocating more memory if the capacity is reached.

The following code shows how to inserts a value into a vector at position $5$, provided that the vector has at least $6$ elements:

```cpp
if (vec.size() > 5) {
    vec.insert(vec.begin() + 5, 32);
}
```

`vec.insert(vec.begin() + 5, 42);` inserts the value $32$ at position $5$ in the vector. The condition `vec.size() > 5` ensures the vector has enough elements to avoid out-of-bounds errors.

```cpp
if (vec.size() > 5) vec.insert(vec.begin() + 5, 42);
```

By removing the block braces `{}`, the code becomes more concise while keeping the logic intact. This is useful when clarity is maintained without additional syntax.

Using `#define` for Typing Efficiency:

```cpp
#define VI std::vector<int>
VI vec;
if (vec.size() > 5) vec.insert(vec.begin() + 5, 42);
```

The `#define` statement creates a shorthand for `std::vector<int>`, reducing typing. `VI vec;` now declares the vector `vec` as a vector of integers. The rest of the logic remains unchanged, making the code easier to write without losing clarity.

Insert is not enough. The following code removes the last element from the vector, followed by the removal of the element at position $3$, assuming the vector has at least $4$ elements:

```cpp
if (!vec.empty()) {
    vec.pop_back();
}

if (vec.size() > 3) {
    vec.erase(vec.begin() + 3);
}
```

`vec.pop_back();` removes the last element from the vector while `vec.erase(vec.begin() + 3);` removes the element at position $3$. Using predefined macros, we can also reduce typing for common operations:

```cpp
#define ERASE_AT(vec, pos) vec.erase(vec.begin() + pos)
if (!vec.empty()) vec.pop_back();
if (vec.size() > 3) ERASE_AT(vec, 3);
```

We also can create vectors with a default value in all positions. The following code creates a new integer vector with $5$ elements, all initialized to the value $7$:

```cpp
std::vector<int> vec2(5, 7);
```

No significant reduction can be achieved here without compromising clarity, but using `#define` can help in repetitive vector use situations or multiple operations:

```cpp
#define VI std::vector<int>
VI vec2(5, 7);
```

In the next code fragment, a more complex example, the vector `vec2` is resized to $10$ elements, and each element is filled with a random value between $1$ and $100$:

```cpp
vec2.resize(10);

unsigned seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

std::mt19937 generator(seed);
std::uniform_int_distribution<int> distribution(1, 100);

for (size_t i = 0; i < vec2.size(); ++i) {
    vec2[i] = distribution(generator);
}
```

The line `vec2.resize(10);` changes the size of the vector to hold $10$ elements. If the vector had fewer than $10$ elements, new ones are added. If it had more, extra elements are removed.

Next, the line `unsigned seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count());` creates a seed for the random number generator. It uses the current time, converts it to a count of time units, and casts it to an unsigned integer. This ensures the random numbers change each time the program runs.

The code then defines the generator `std::mt19937 generator(seed);` using the seed. This creates a Mersenne Twister random number generator, which is fast and reliable. The next line, `std::uniform_int_distribution<int> distribution(1, 100);`, sets up the distribution. It ensures random numbers between 1 and 100 are generated evenly.

Finally, the `for` loop runs through each element in the vector. It assigns a random number to each position, filling the vector with random values between $1$ and $100$. We can rewrite this same code with less typing.

```cpp
vec2.resize(10);
std::mt19937 gen(std::random_device{}());
std::uniform_int_distribution<int> dist(1, 100);
for (auto& v : vec2) v = dist(gen);
```

> In C++20, we use the `<random>` library for generating random numbers. This library gives us flexible and efficient ways to create random data. It includes generators and distributions to produce random numbers in various forms.
>
> We start with `std::mt19937`, which is a Mersenne Twister generator. This generator is known for being fast and producing high-quality random numbers. It's not new to C++20, but it’s still a strong choice. The generator works by taking a seed, usually an integer, to initialize its random sequence. In the code, we use:
>
> ```cpp
> std::mt19937 generator(seed);
> ```
>
> Here, `seed` is an unsigned integer. It ensures that every run of the program generates different numbers. We seed the generator using the current time from `std::chrono::high_resolution_clock::now()`, converted into an unsigned integer. This makes the sequence unpredictable.
>
> The generator itself only produces random bits. To convert those bits into a useful range, we use a distribution. C++20 offers several kinds of distributions. For example, we use `std::uniform_int_distribution<int>`, which produces integers evenly spread across a given range:
>
> ```cpp
> std::uniform_int_distribution<int> distribution(1, 100);
> ```
>
> This tells the program to create numbers between $1$ and $100$, all with equal probability. When we call:
>
> ```cpp
> distribution(generator);
> ```
>
> The generator provides the random bits, and the distribution maps those bits to numbers between $1$ and $100$.
>
> C++20 keeps these ideas separate: generators produce bits, and distributions map those bits to useful values. You can also use other distributions like `std::normal_distribution` for normal (Gaussian) distributions or `std::bernoulli_distribution` for true/false outcomes.
>
> If you need more control, C++20 introduces new features like `std::seed_seq` for better seeding and >`std::random_device` for non-deterministic seeds. But for most competitive programming tasks, the Mersenne >Twister and a simple distribution will do the job well. You can see a example in the following code:
>
> ```cpp
> #include <iostream>
> #include <vector>
> #include <random>
> #include <chrono>
>
> int main() {
>     // Step 1: Initialize a vector with 10 elements
>     std::vector<int> vec;
>     vec.resize(10); // Resizing the vector to hold 10 elements
>
>     // Step 2: Use std::random_device for non-deterministic seeding
>     std::random_device rd; // Non-deterministic random seed generator
>
>     // Step 3: Use std::seed_seq for better seeding control
>     std::seed_seq seed{rd(), rd(), rd(), rd(), rd()}; // Collect multiple seeds
>     std::mt19937 generator(seed); // Mersenne Twister seeded with the seed sequence
>
>     // Step 4: Create a uniform distribution to generate integers between 1 and 100
>     std::uniform_int_distribution<int> distribution(1, 100);
>
>     // Step 5: Fill the vector with random integers using the generator and distribution
>     for (size_t i = 0; i < vec.size(); ++i) {
>         vec[i] = distribution(generator); // Assign a random value to each element
>     }
>
>     // Step 6: Print the randomly generated numbers in the vector
>     std::cout << "Random numbers in the vector: ";
>     for (const auto& num : vec) {
>         std::cout << num << " "; // Output each random number
>     }
>     std::cout << std::endl;
>
>     return 0;
> }
> ```
>
> The key to using random numbers in C++20 is understanding the separation of concerns: generators make the random >bits, and distributions convert those bits into meaningful values. This approach makes the system flexible and >powerful.

### 7.1.5 Sorting the Vector

The following code sorts the vector `vec2` in ascending order:

**Standard Version:**

```cpp
std::sort(vec2.begin(), vec2.end());
```

- $ \text{std::sort(vec2.begin(), vec2.end());} $ sorts the vector in ascending order.

\*### Optimized for Minimal Typing with `constexpr`

We can replace the `#define` with a `constexpr` function, which provides type safety and integrates better with the C++ type system.

**Using `constexpr` for Sorting a Vector:**

```cpp
constexpr void sort_vector(std::vector<int>& vec) {
    std::sort(vec.begin(), vec.end());
}

sort_vector(vec2);
```

- $ \text{constexpr void sort_vector(std::vector<int}\& vec)$ is a type-safe way to define a reusable sorting function.
- This method avoids the pitfalls of `#define`, such as lack of scoping and type checking, while still minimizing the amount of typing.

### 7.1.6 Vectors as Inputs and Outputs

In competitive programming, a common input format involves receiving the size of a vector as the first integer, followed by the elements of the vector separated by spaces, with a newline at the end. Handling this efficiently is crucial when dealing with large inputs. Below is an optimized version using `fread` for input and `putchar` for output, ensuring minimal system calls and fast execution.

This version reads the input, processes it, and then outputs the vector’s elements using the fastest possible I/O methods in C++.

```cpp
#include <cstdio>
#include <vector>

int main() {
    // Buffer for reading input
    char buffer[1 << 16]; // 64 KB buffer size
    int idx = 0;

    // Read the entire input at once
    size_t bytesRead = fread(buffer, 1, sizeof(buffer), stdin);

    // Parse the size of the vector from the input
    int n = 0;
    while (buffer[idx] >= '0' && buffer[idx] <= '9') {
        n = n * 10 + (buffer[idx++] - '0');
    }
    ++idx; // Skip the space or newline after the number

    // Create the vector and fill it with elements
    std::vector<int> vec(n);
    for (int i = 0; i < n; ++i) {
        int num = 0;
        while (buffer[idx] >= '0' && buffer[idx] <= '9') {
            num = num * 10 + (buffer[idx++] - '0');
        }
        vec[i] = num;
        ++idx; // Skip the space or newline after each number
    }

    // Output the vector elements using putchar
    for (int i = 0; i < n; ++i) {
        if (vec[i] == 0) putchar('0');
        else {
            int num = vec[i], digits[10], digitIdx = 0;
            while (num) {
                digits[digitIdx++] = num % 10;
                num /= 10;
            }
            // Print digits in reverse order
            while (digitIdx--) putchar('0' + digits[digitIdx]);
        }
        putchar(' '); // Space after each number
    }
    putchar('\n'); // End the output with a newline

    return 0;
}
```

In the previous code, we have the following functions:

1. **Input with `fread`**:
   - `fread` is used to read the entire input into a large buffer at once. This avoids multiple system calls, which are slower than reading in bulk.
2. **Parsing the Input**:
   - The input is parsed from the buffer using simple character arithmetic to convert the string of numbers into integers.
3. **Output with `putchar`**:
   - `putchar` is used to print the numbers, which is faster than `std::cout` for individual characters. The digits of each number are processed and printed in reverse order.

The previous code method minimizes system calls and avoids using slower I/O mechanisms like `std::cin` and `std::cout`, making it highly optimized for competitive programming scenarios where speed is crucial.

In competitive programming, it's also common to handle input from a file provided via the command line. This scenario requires efficient reading and processing, especially when dealing with large datasets. Below is the optimized version using `fread` to read from a file specified in the command line argument and `putchar` for output.

#### 7.1.6.1 Optimized Version Using `fread` and `putchar` with Command-Line File Input

This version reads the input file, processes it, and outputs the vector’s elements, ensuring fast I/O performance.

```cpp
#include <cstdio>
#include <vector>

int main(int argc, char* argv[]) {
    // Check if the filename was provided
    if (argc != 2) {
        return 1;
    }

    // Open the file from the command line argument
    FILE* file = fopen(argv[1], "r");
    if (!file) {
        return 1;
    }

    // Buffer for reading input
    char buffer[1 << 16]; // 64 KB buffer size
    int idx = 0;

    // Read the entire input file at once
    size_t bytesRead = fread(buffer, 1, sizeof(buffer), file);
    fclose(file); // Close the file after reading

    // Parse the size of the vector from the input
    int n = 0;
    while (buffer[idx] >= '0' && buffer[idx] <= '9') {
        n = n * 10 + (buffer[idx++] - '0');
    }
    ++idx; // Skip the space or newline after the number

    // Create the vector and fill it with elements
    std::vector<int> vec(n);
    for (int i = 0; i < n; ++i) {
        int num = 0;
        while (buffer[idx] >= '0' && buffer[idx] <= '9') {
            num = num * 10 + (buffer[idx++] - '0');
        }
        vec[i] = num;
        ++idx; // Skip the space or newline after each number
    }

    // Output the vector elements using putchar
    for (int i = 0; i < n; ++i) {
        if (vec[i] == 0) putchar('0');
        else {
            int num = vec[i], digits[10], digitIdx = 0;
            while (num) {
                digits[digitIdx++] = num % 10;
                num /= 10;
            }
            // Print digits in reverse order
            while (digitIdx--) putchar('0' + digits[digitIdx]);
        }
        putchar(' '); // Space after each number
    }
    putchar('\n'); // End the output with a newline

    return 0;
}
```

In the previous code we have:

1. **File Input with `fread`**:

   - The input is read from a file specified in the command line argument using `fread`. This reads the entire file into a buffer in one go, improving efficiency by reducing system calls.

2. **File Handling**:

   - The file is opened using `fopen` and closed immediately after reading the data. This ensures that system resources are released as soon as the file reading is complete.

3. **Parsing and Output**:
   - The rest of the program processes the input similarly to the previous version, parsing the numbers from the buffer and outputting them efficiently using `putchar`.

This approach remains highly optimized for competitive programming environments where fast I/O handling is critical. But, in Linux we can use `mmap`

```cpp
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }

    // Open the file
    int fd = open(argv[1], O_RDONLY);
    if (fd == -1) {
        return 1;
    }

    // Get the file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        return 1;
    }
    size_t fileSize = sb.st_size;

    // Memory-map the file
    char* fileData = static_cast<char*>(mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0));
    if (fileData == MAP_FAILED) {
        close(fd);
        return 1;
    }

    close(fd); // The file descriptor can be closed after mapping

    // Parse the vector size
    int idx = 0;
    int n = 0;
    while (fileData[idx] >= '0' && fileData[idx] <= '9') {
        n = n * 10 + (fileData[idx++] - '0');
    }
    ++idx; // Skip the space or newline

    // Create the vector and fill it with values from the memory-mapped file
    std::vector<int> vec(n);
    for (int i = 0; i < n; ++i) {
        int num = 0;
        while (fileData[idx] >= '0' && fileData[idx] <= '9') {
            num = num * 10 + (fileData[idx++] - '0');
        }
        vec[i] = num;
        ++idx; // Skip the space or newline
    }

    // Output the vector
    for (const int& num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Unmap the file from memory
    munmap(fileData, fileSize);

    return 0;
}
```

## 7.2 Matrices

In C++20, matrices are typically represented as vectors of vectors (`std::vector<std::vector<T>>`), where each inner vector represents a row of the matrix. This approach allows for dynamic sizing and easy manipulation of multi-dimensional data, making matrices ideal for problems involving grids, tables, or any 2D structure.

Matrices in C++ offer flexibility in managing data: you can resize rows and columns independently, access elements using intuitive indexing, and leverage standard vector operations for rows. Additionally, the use of `ranges` and `views` introduced in C++20 boosts the ability to iterate and transform matrix data more expressively and efficiently.

_The use of matrices is common in competitive programming for tasks such as implementing dynamic programming tables, graph adjacency matrices, or performing transformations on 2D data. With the powerful capabilities of C++20's STL, matrices become a highly adaptable and efficient way to handle complex, multi-dimensional computations in a structured manner_.

### 7.2.1 Creating and Filling a Matrix

The code creates a 2x2 matrix (a vector of vectors) and fills each element with the value 1:

**Standard Version:**

```cpp
int rows = 2, cols = 2;
std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
        matrix[i][j] = 1;
    }
}
```

- $ \text{std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));} $ creates a matrix of size $2\times 2$.
- The nested `for` loop fills each element of the matrix with $1$.

**Optimized for Minimal Typing:**

```cpp
std::vector<std::vector<int>> matrix(2, std::vector<int>(2, 1));
```

This version eliminates the need for the explicit loop by using the constructor to initialize the matrix with 1s directly.

### 7.2.2 Displaying the Matrix

Finally, the matrix is printed in the standard format:

**Standard Version:**

```cpp
for (const auto& row : matrix) {
    for (const auto& element : row) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}
```

- The loop iterates over each row and prints all elements in the row, followed by a newline.

**Optimized for Minimal Typing:**

```cpp
for (const auto& row : matrix) {
    for (int el : row) std::cout << el << " ";
    std::cout << "\n";
}
```

Here, we replaced `std::endl` with `"\n"` to improve performance by avoiding the unnecessary flushing of the output buffer.

### 7.2.3 Inserting Elements at a Specific Position

To insert an element at a specific position in a matrix (vector of vectors) in C++ 20, we use the `insert` function. This function can insert rows or columns in a specific location, modifying the structure of the matrix.

```cpp

#include <iostream>
#include <vector>

int main() {
    std::vector<std::vector<int>> matrix = { {1, 2}, {3, 4} };

    // Insert a row at position 1
    matrix.insert(matrix.begin() + 1, std::vector<int>{5, 6});

    // Insert a column value at position 0 in the first row
    matrix[0].insert(matrix[0].begin(), 0);

    // Display the modified matrix
    for (const auto& row : matrix) {
        for (int el : row) std::cout << el << " ";
        std::cout << "\n";
    }

    return 0;
}
```

This code inserts a new row at position 1 and a new column value at position 0 in the first row. The result is a modified matrix.

### 7.2.4 Removing the Last Element and a Specific Element

To remove the last element of a matrix or a specific element, you can use the `pop_back` function for removing the last row and the `erase` function for removing specific rows or columns.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<std::vector<int>> matrix = { {1, 2}, {3, 4}, {5, 6} };

    // Remove the last row
    matrix.pop_back();

    // Remove the first element of the first row
    matrix[0].erase(matrix[0].begin());

    // Display the modified matrix
    for (const auto& row : matrix) {
        for (int el : row) std::cout << el << " ";
        std::cout << "\n";
    }

    return 0;
}
```

This code removes the last row from the matrix and removes the first element of the first row.

### 7.2.5 Creating a New Vector with a Default Value

To create a new matrix filled with a default value, you can specify this value in the constructor of the vector.

```cpp
#include <iostream>
#include <vector>

int main() {
    // Create a 3x3 matrix filled with the default value 7
    std::vector<std::vector<int>> matrix(3, std::vector<int>(3, 7));

    // Display the matrix
    for (const auto& row : matrix) {
        for (int el : row) std::cout << el << " ";
        std::cout << "\n";
    }

    return 0;
}
```

This code initializes a 3x3 matrix with all elements set to 7.

### 7.2.6 Resizing and Filling with Random Values

To resize a matrix and fill it with random values, you can use the `resize` function along with the `<random>` library.

```cpp
#include <iostream>
#include <vector>
#include <random>

int main() {
    std::vector<std::vector<int>> matrix;
    int rows = 3, cols = 3;

    // Resize the matrix
    matrix.resize(rows, std::vector<int>(cols));

    // Fill the matrix with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (auto& row : matrix) {
        for (auto& el : row) {
            el = dis(gen);
        }
    }

    // Display the matrix
    for (const auto& row : matrix) {
        for (int el : row) std::cout << el << " ";
        std::cout << "\n";
    }

    return 0;
}
```

This code resizes the matrix to 3x3 and fills it with random values between 1 and 10.

### 7.2.7 Sorting Matrices by Rows and Columns

In C++20, we can sort matrices (represented as vectors of vectors) both by rows and by columns. Here are examples of how to do both:

#### 7.2.7.1 Sorting by Rows

Sorting by rows is straightforward, as we can use the `std::sort` function directly on each row of the matrix.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
   std::vector<std::vector<int>> matrix = {
        {3, 1, 4}, {1, 5, 9}, {2, 6, 5}
    };

   // Sort each row of the matrix
   for (auto& row : matrix) {
       std::sort(row.begin(), row.end());
   }

   // Display the sorted matrix
   for (const auto& row : matrix) {
       for (int el : row) std::cout << el << " ";
       std::cout << "\n";
   }

   return 0;
}
```

This code sorts each row of the matrix independently. The time complexity for sorting by rows is $O(m \cdot n \log n)$, where $m$ is the number of rows and $n$ is the number of columns.

#### 7.2.7.2 Sorting by Columns

Sorting by columns is more complex because the elements in a column are not contiguous in memory. We need to extract each column, sort it, and then put the sorted elements back into the matrix.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
   std::vector<std::vector<int>> matrix = { {3, 1, 4}, {1, 5, 9}, {2, 6, 5} };
   int rows = matrix.size();
   int cols = matrix[0].size();

   // Sort each column of the matrix
   for (int j = 0; j < cols; ++j) {
       std::vector<int> column;
       for (int i = 0; i < rows; ++i) {
           column.push_back(matrix[i][j]);
       }
       std::sort(column.begin(), column.end());
       for (int i = 0; i < rows; ++i) {
           matrix[i][j] = column[i];
       }
   }

   // Display the sorted matrix
   for (const auto& row : matrix) {
       for (int el : row) std::cout << el << " ";
       std::cout << "\n";
   }

   return 0;
}
```

This code sorts each column of the matrix independently. The time complexity for sorting by columns is $O(n \cdot m \log m)$, where $n$ is the number of columns and $m$ is the number of rows.

Note that this method of sorting by columns is not the most efficient for very large matrices, as it involves many data copies. For large matrices, it might be more efficient to use an approach that sorts the row indices based on the values in a specific column.

### 7.2.8 Optimizing Matrix Input and Output in Competitive Programming

In competitive programming, efficiently handling matrices for input and output is crucial. Let's explore optimized techniques in C++ that minimize system calls and maximize execution speed.

Typically, the input for a matrix consists of:

1. Two integers $n$ and $m$, representing the number of rows and columns, respectively.
2. $n \times m$ elements of the matrix, separated by spaces and newlines.

For example:

```txt
3 4
1 2 3 4
5 6 7 8
9 10 11 12
```

### 7.2.8.1 Optimized Reading with `fread`

To optimize reading, we can use `fread` to load the entire input at once into a buffer, then parse the numbers from the buffer. This approach reduces the number of system calls compared to reading the input one character or one line at a time.

```cpp
#include <cstdio>
#include <vector>

int main() {
    char buffer[1 << 16];
    size_t bytesRead = fread(buffer, 1, sizeof(buffer), stdin);
    size_t idx = 0;

    auto readInt = [&](int& num) {
        while (idx < bytesRead && (buffer[idx] < '0' || buffer[idx] > '9') && buffer[idx] != '-') ++idx;
        bool neg = false;
        if (buffer[idx] == '-') {
            neg = true;
            ++idx;
        }
        num = 0;
        while (idx < bytesRead && buffer[idx] >= '0' && buffer[idx] <= '9') {
            num = num * 10 + (buffer[idx++] - '0');
        }
        if (neg) num = -num;
    };

    int n, m;
    readInt(n);
    readInt(m);

    std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            readInt(matrix[i][j]);
        }
    }

    // Matrix processing...

    return 0;
}
```

In this code: We define a lambda function `readInt` to read integers from the buffer, handling possible whitespace and negative numbers. The `readInt` function skips over any non-digit characters and captures negative signs. This ensures robust parsing of the input data.

### 7.2.8.2 Optimized Output with `putchar_unlocked`

For output, using `putchar_unlocked` offers better performance than `std::cout` or even `putchar`, as it is not thread-safe and thus faster.

```cpp
#include <cstdio>
#include <vector>

void writeInt(int num) {
    if (num == 0) {
        putchar_unlocked('0');
        return;
    }
    if (num < 0) {
        putchar_unlocked('-');
        num = -num;
    }
    char digits[10];
    int idx = 0;
    while (num) {
        digits[idx++] = '0' + num % 10;
        num /= 10;
    }
    while (idx--) {
        putchar_unlocked(digits[idx]);
    }
}

int main() {
    // Assume matrix is already populated
    int n = /* number of rows */;
    int m = /* number of columns */;
    std::vector<std::vector<int>> matrix = /* your matrix */;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            writeInt(matrix[i][j]);
            putchar_unlocked(j == m - 1 ? '\n' : ' ');
        }
    }

    return 0;
}
```

In this code: We define a function `writeInt` to output integers efficiently. It handles zero and negative numbers correctly, and we use `putchar_unlocked` for faster character output.

> `putchar_unlocked` is a non-thread-safe version of `putchar`. It writes a character to `stdout` without locking the output stream, eliminating the overhead associated with ensuring thread safety. This makes `putchar_unlocked` faster than `putchar`, which locks the output stream to prevent concurrent access from multiple threads.
>
> When comparing `putchar` and `putchar_unlocked`, we find that `putchar` is thread-safe and locks `stdout` to prevent data races, but incurs overhead due to locking. On the other hand, `putchar_unlocked` is not thread-safe and does not lock `stdout`, making it faster due to the absence of locking overhead.
>
> Here's an example of using `putchar_unlocked` to output an integer efficiently:
>
> ```cpp
> #include <cstdio>
>
> void writeInt(int num) {
>    if (num == 0) {
>        putchar_unlocked('0');
>        return;
>    }
>    if (num < 0) {
>        putchar_unlocked('-');
>        num = -num;
>    }
>    char digits[10];
>    int idx = 0;
>    while (num) {
>        digits[idx++] = '0' + (num % 10);
>        num /= 10;
>    }
>    while (idx--) {
>        putchar_unlocked(digits[idx]);
>    }
> }
>
> int main() {
>    int number = 12345;
>    writeInt(number);
>    putchar_unlocked('\n');
>    return 0;
> }
> ```
>
> In contrast, using `putchar` would involve replacing `putchar_unlocked` with `putchar`:
>
> ```cpp
> #include <cstdio>
>
> void writeInt(int num) {
>    if (num == 0) {
>        putchar('0');
>        return;
>    }
>    if (num < 0) {
>        putchar('-');
>        num = -num;
>    }
>    char digits[10];
>    int idx = 0;
>    while (num) {
>        digits[idx++] = '0' + (num % 10);
>        num /= 10;
>    }
>    while (idx--) {
>        putchar(digits[idx]);
>    }
> }
>
> int main() {
>    int number = 12345;
>    writeInt(number);
>    putchar('\n');
>    return 0;
> }
> ```
>
> `putchar_unlocked` is best used in single-threaded programs where maximum output performance is required. It's particularly >useful in competitive programming scenarios where execution time is critical and the program is guaranteed to be >single-threaded.
>
> However, caution must be exercised when using `putchar_unlocked`. It is not thread-safe, and in multi-threaded applications, >using it can lead to data races and undefined behavior. Additionally, it is a POSIX function and may not be available or >behave differently on non-POSIX systems.
>
> _Both `putchar` and `putchar_unlocked` are functions from the C standard library `<cstdio>`, which is included in C++ for >compatibility purposes. The prototype for `putchar` is `int putchar(int character);`, which writes the character to `stdout` >and returns the character written, or `EOF` on error. It is thread-safe due to internal locking mechanisms_.
>
> The prototype for `putchar_unlocked` is `int putchar_unlocked(int character);`. It's a faster version of `putchar` without >internal locking, but it's not thread-safe and may not be part of the C++ standard in all environments.
>
> If both performance and thread safety are needed, consider using buffered output or high-performance C++ I/O techniques. For >example:
>
> ```cpp
> #include <iostream>
> #include <vector>
>
> int main() {
>    std::ios::sync_with_stdio(false);
>    std::cin.tie(nullptr);
>
>    std::vector<int> numbers = {1, 2, 3, 4, 5};
>    for (int num : numbers) {
>        std::cout << num << ' ';
>    }
>    std::cout << '\n';
>
>    return 0;
> }
> ```
>
> By untethering C++ streams from C streams using `std::ios::sync_with_stdio(false);` and untangling `cin` from `cout` with >`std::cin.tie(nullptr);`, you can achieve faster I/O while maintaining thread safety and standard compliance.

### 7.2.8.3 Complexity Analysis

The time complexity for reading and writing is $O(nm)$, where $n$ and $m$ are the dimensions of the matrix. The space complexity is also $O(nm)$, as we store the entire matrix in memory. However, the constant factors are significantly reduced compared to standard I/O methods, leading to faster execution times in practice.

### 7.2.8.4 Using `mmap` on Unix Systems

On Unix systems, we can use `mmap` to map a file (or standard input) directly into memory, potentially improving I/O performance even further.

```cpp
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <cstdio>

int main() {
    struct stat sb;
    fstat(0, &sb); // File descriptor 0 is stdin
    size_t fileSize = sb.st_size;
    char* data = static_cast<char*>(mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, 0, 0));

    size_t idx = 0;

    auto readInt = [&](int& num) {
        while (idx < fileSize && (data[idx] < '0' || data[idx] > '9') && data[idx] != '-') ++idx;
        bool neg = false;
        if (data[idx] == '-') {
            neg = true;
            ++idx;
        }
        num = 0;
        while (idx < fileSize && data[idx] >= '0' && data[idx] <= '9') {
            num = num * 10 + (data[idx++] - '0');
        }
        if (neg) num = -num;
    };

    int n, m;
    readInt(n);
    readInt(m);

    std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            readInt(matrix[i][j]);
        }
    }

    munmap(data, fileSize);

    // Matrix processing...

    return 0;
}
```

_**Note:** Using `mmap` can be risky, as it relies on the entire input being available and may not be portable across different systems or handle input streams properly. Use it only when you are certain of the input's nature and when maximum performance is essential._

_**Remember:** The efficiency of these approaches comes at the cost of increased code complexity and reduced readability. In scenarios where performance is not critical, standard I/O methods are preferable for their simplicity and maintainability._

## 7.3 Efficient Data Manipulation in C++ using Span and Ranges

In the fast-paced world of competitive programming and high-performance computing, efficient data manipulation is paramount. C++20 introduces two powerful features - `std::span` and `std::ranges` for that.

These features are particularly important because they address common performance bottlenecks in data-intensive applications. `std::span` provides a lightweight, non-owning view into contiguous data, reducing unnecessary copying and allowing for flexible, efficient data access. `std::ranges`, on the other hand, offers a unified, composable interface for working with sequences of data, enabling more intuitive and often more performant algorithm implementations. Together, they form a potent toolkit for developers seeking to push the boundaries of what's possible in terms of code efficiency and elegance in C++.

### 7.3.1 Using `std::span`

The `std::span` is a new feature introduced in C++20 that allows you to create lightweight, non-owning views of arrays and containers, such as `std::vector`. This avoids unnecessary copying of data and provides a flexible and efficient way to access and manipulate large blocks of data. `std::span` can be particularly useful when working with large datasets, file I/O, or when optimizing memory usage in competitive programming.

Unlike containers such as `std::vector`, `std::span` doesn't own the data it references. This means it doesn't allocate new memory and works directly with existing data, leading to lower memory overhead. Additionally, `std::span` can work with both static arrays and dynamic containers (like `std::vector`) without requiring copies. It provides safer array handling compared to raw pointers, as it encapsulates size information. Since `std::span` eliminates the need for memory copies, it can speed up operations where large datasets need to be processed in-place, or only certain views of data are required.

**Example of `std::span` for Efficient Data Access**:

In this example, we create a `std::span` from a `std::vector` of integers, allowing us to iterate over the vector’s elements without copying the data:

```cpp
#include <iostream>
#include <span>
#include <vector>

int main() {
    // Create a vector of integers
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // Create a span view of the vector
    std::span<int> view(numbers);

    // Iterate over the span and print the values
    for (int num : view) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

`std::span<int> view(numbers);` creates a non-owning view of the `std::vector<int>` `numbers`. This allows access to the elements of the vector without copying them. The loop `for (int num : view)` iterates over the elements in the `std::span`, just like it would with the original `std::vector`, but with no additional overhead from copying the data.

#### 7.3.1.1 Efficient Use Cases for `std::span`

`std::span` is especially useful when you want to work with sub-ranges of arrays or vectors. For example, when working with just part of a large dataset, you can use `std::span` to reference a subset without slicing or creating new containers:

```cpp
std::span<int> subrange = view.subspan(1, 3); // Access elements 1, 2, and 3
for (int num : subrange) {
    std::cout << num << " "; // Outputs: 2 3 4
}
```

When passing data to functions, `std::span` provides an efficient alternative to passing large vectors or arrays by reference. You can pass a span instead, ensuring that no copies are made, while maintaining full access to the original data:

```cpp
void process_data(std::span<int> data) {
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> numbers = {10, 20, 30, 40, 50};
    process_data(numbers); // Pass the vector as a span
    return 0;
}
```

In the early example, the function `process_data` accepts a `std::span`, avoiding unnecessary copies and keeping the original data structure intact.

#### 7.3.1.2 Comparing `std::span` to Traditional Methods

| Feature          | `std::vector`           | Raw Pointers          | `std::span`     |
| ---------------- | ----------------------- | --------------------- | --------------- |
| Memory Ownership | Yes                     | No                    | No              |
| Memory Overhead  | High (allocates memory) | Low                   | Low             |
| Bounds Safety    | High                    | Low                   | High            |
| Compatibility    | Works with STL          | Works with raw arrays | Works with both |

Unlike `std::vector`, which manages its own memory, `std::span` does not allocate or own memory. This is similar to raw pointers but with added safety since `std::span` knows its size. `std::span` is safer than raw pointers because it carries bounds information, helping avoid out-of-bounds errors. While raw pointers offer flexibility, they lack the safety features provided by modern C++.

#### 7.3.1.3 Practical Application: Using `std::span` in Competitive Programming

When working with large datasets in competitive programming, using `std::span` avoids unnecessary memory copies, making operations faster and more efficient. You can easily pass sub-ranges of data to functions without creating temporary vectors or arrays. Additionally, it allows you to maintain full control over memory without introducing complex ownership semantics, as with `std::unique_ptr` or `std::shared_ptr`.

**Example: Efficiently Passing Data in a Competitive Programming Scenario**:

```cpp
#include <iostream>
#include <span>
#include <vector>

void solve(std::span<int> data) {
    for (int num : data) {
        std::cout << num * 2 << " "; // Example: print double each value
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> input = {100, 200, 300, 400, 500};

    // Use std::span to pass the entire vector without copying
    solve(input);

    // Use a subspan to pass only a portion of the vector
    solve(std::span<int>(input).subspan(1, 3)); // Pass elements 200, 300, 400

    return 0;
}
```

### 7.4 Efficient Data Manipulation with `std::ranges` in C++20

C++20 brought the `<ranges>` library—a tool for handling data sequences with power and ease. It works through lazy views and composable transformations. `std::ranges` lets you create views over containers or arrays, avoiding changes and extra copies. In competitive programming and high-performance tasks, cutting down on memory and computation is key.

With `std::vector` and other containers, you often need extra storage or loops for things like filtering, transforming, or slicing data. `std::ranges` changes that. It lets you chain operations in a simple, expressive way without losing speed. It uses lazy evaluation, meaning transformations only happen when needed, not upfront.

`std::ranges` revolves around "views" of data, a windows that let you look at and manipulate sequences without owning them. A view acts like a container, but it doesn't hold the data itself. It just provides a way to interact with it, making operations light and efficient.

The advantage? `std::ranges` stacks operations without creating new containers, saving memory. Traditional methods create copies with every action (filter, transform, slice) adding overhead. Ranges avoid this, evaluating only when the data is accessed. Memory stays low, performance stays high, especially with big data.

Performance gains also come from optimized operations. By working lazily and directly on the data, `std::ranges` avoids unnecessary copies and allocations. The result is better cache usage and fewer CPU cycles wasted on managing temporary structures.

**Example Filtering and Transforming Data with `std::ranges`**:

Suppose we have a vector of integers and we want to filter out the odd numbers and then multiply the remaining even numbers by two. Using traditional methods, we would need to loop through the vector, apply conditions, and store the results in a new container. With `std::ranges`, this can be done in a more expressive and efficient way:

```cpp
#include <iostream>
#include <vector>
#include <ranges>
#include <span>

int main() {
    // Sample data
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Using std::span to create a view over the existing data without copying
    std::span<int> data_span(data);

    // Using std::ranges to filter and transform data lazily
    auto processed_view = data_span
        | std::views::filter([](int x) { return x % 2 == 0; }) // Filter even numbers
        | std::views::transform([](int x) { return x * x; });  // Square each number

    // Iterating over the processed view
    for (int value : processed_view) {
        std::cout << value << " ";
    }

    return 0;
}

```

The line that creates `processed_view` does the heavy lifting. It uses `std::span` and `std::ranges` to work smart, not hard. Here's what happens, step by step:

```cpp
auto processed_view = data_span
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; });
```

First, `data_span` is a direct window into the data. No copies, no waste. Then comes `std::views::filter`. It uses a lambda function, which is just a fancy name for a tiny, anonymous function that you write right there. The filter is simple—it checks each number, `[] (int x) { return x % 2 == 0; }`. It says, "If the number is even, keep it; if not, toss it." No fuss.

Next is `std::views::transform`. It’s another lambda, `[] (int x) { return x * x; }`. It takes what’s left from the filter and squares each number. The job is done on the fly, no waiting. The power here is that everything happens only when needed. No intermediate results, no unnecessary containers. Just efficient, on-demand calculation. The functions do their work with precision—like a scalpel, not a sledgehammer.

> The `|` operator in C++20 is called the pipe operator. It chains ranges together, like pieces of a machine, one feeding into the next, just like a Unix pipeline. In the `processed_view` line, it links `std::views::filter` first, then `std::views::transform`. Data moves step by step, each part doing its job without extra fuss. No need for temporary variables or extra code. It’s clean, efficient. Each transformation builds on the last, clear as a straight line.
>
> But the pipe doesn’t stop with ranges. You can use it with custom classes or user types too. Overload the operator, and you’ve got yourself a clear chain of actions, each link precise and sharp. It can take complex operations and make them simple, one step feeding the next, easy to read and hard to get wrong.
>
> The pipe works with I/O too. It lets you stack stream manipulators like a craftsman arranging his tools,`std::cout | std::hex | std::uppercase`. It’s smooth, no clutter. In functional code, pipes connect functions, turning data flow into a straight path. They make the code tell a story—one step at a time, each part pulling its weight.
>
> The `|` operator isn’t just a trick for ranges; it’s a way to keep the code honest. It turns complex work into a direct line, clear, readable, and true to the task.

One of the main strengths of `std::ranges` is how you can stack operations, one on top of the other. You filter, you transform, you slice—and it all stays as a view. No extra containers, no wasted steps. You build a pipeline that works only when you call on it, no sooner. It’s efficient and lean, cutting through the data like a sharp knife. You get what you need when you need it, nothing more. The code is clean, each piece doing its job, each step feeding the next without clutter or delay.

Consider another example, where we filter, transform, and take only a part of the data:

```cpp
#include <iostream>
#include <vector>
#include <ranges>
#include <span>

int main() {
    // Sample data
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Using std::span to create a direct, non-owning view of the data
    std::span<int> data_span(data);

    // Using std::ranges with std::span to filter, transform, and limit the data efficiently
    auto processed_view = data_span
        | std::views::filter([](int x) { return x % 2 == 0; })  // Filter even numbers
        | std::views::transform([](int x) { return x * x; })   // Square each number
        | std::views::take(3);                                // Take the first 3 elements

    // Iterate over the processed view
    for (int value : processed_view) {
        std::cout << value << " ";
    }

    return 0;
}
```

In this example, we stack three operations. First, we filter the numbers, keeping those $20$ or higher. Then, we double them. Finally, we take the first three. It all happens lazily, nothing done until you need it. When you loop through the final view, `result`, that's when the work gets done. No extra containers, no wasted steps. Each transformation hits just once, right where it counts. The code stays lean, the processing sharp and efficient, doing just enough—no more, no less.
