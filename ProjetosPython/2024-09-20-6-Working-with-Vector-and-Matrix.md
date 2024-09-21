---
author: Frank
beforetoc: '[Anterior](2024-09-20-5-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-7-Sem-T%C3%ADtulo.md)'
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
title: Working with Vector and Matrix
toc: true
---
# Working with Vector and Matrix

Vectors are one of the most versatile data structures used in competitive programming due to their dynamic size and ease of use. They allow for efficient insertion, removal, resizing, and access operations, making them suitable for a wide range of applications. Not only can vectors handle single-dimensional data, but they can also represent more complex structures, such as matrices (2D vectors), which are often used to solve grid-based problems, dynamic table calculations, or simulations of multi-dimensional data.

Matrices, represented as vectors of vectors, are particularly useful in problems involving multi-dimensional data manipulation, such as game boards, adjacency matrices in graphs, and dynamic programming tables. Vectors and matrices enable frequent operations like row and column manipulation, matrix transposition, and access to specific submatrices, providing flexibility and control over data arrangement and processing.

## Vectors

In C++, the `vector` class, part of the Standard Template Library (STL), is a dynamic array that provides a versatile and efficient way to manage collections of elements. Unlike traditional arrays, vectors can automatically resize themselves when elements are added or removed, making them particularly useful in competitive programming where the size of data structures may vary during execution.

Vectors offer several advantages: they provide random access to elements, support iteration with iterators, and allow dynamic resizing, which is crucial for managing datasets of unknown or varying lengths. They also support a range of built-in functions for modifying the collection, such as `push_back`, `pop_back`, `insert`, `erase`, and `resize`, allowing developers to manage data efficiently without needing to manually handle memory allocations.

The `vector` class is particularly useful in scenarios involving frequent insertions, deletions, or resizes, as well as when working with dynamic data structures like lists, queues, stacks, or even matrices (2D vectors). Its simplicity and flexibility make it an indispensable tool for implementing a wide range of algorithms quickly and effectively in C++.

### Inserting Elements at a Specific Position

This code inserts a value into a vector at position 5, provided that the vector has at least 6 elements:

**Standard Version:**

```cpp
if (vec.size() > 5) {
    vec.insert(vec.begin() + 5, 42);
}
```

- $ \text{vec.insert(vec.begin() + 5, 42);} $ inserts the value 42 at position 5 in the vector.

**Optimized for Minimal Typing:**

```cpp
if (vec.size() > 5) vec.insert(vec.begin() + 5, 42);
```

By removing the block braces $\{\}$, the code remains concise but still clear in cases where simplicity is essential. Alternatively, you can use the `#define` trick:

```cpp
#define VI std::vector<int>
VI vec;
if (vec.size() > 5) vec.insert(vec.begin() + 5, 42);
```

### Removing the Last Element and a Specific Element

The following code removes the last element from the vector, followed by the removal of the element at position 3, assuming the vector has at least 4 elements:

**Standard Version:**

```cpp
if (!vec.empty()) {
    vec.pop_back();
}

if (vec.size() > 3) {
    vec.erase(vec.begin() + 3);
}
```

- $ \text{vec.pop_back();} $ removes the last element from the vector.
- $ \text{vec.erase(vec.begin() + 3);} $ removes the element at position 3.

**Optimized for Minimal Typing:**

```cpp
if (!vec.empty()) vec.pop_back();
if (vec.size() > 3) vec.erase(vec.begin() + 3);
```

Using predefined macros, we can also reduce typing for common operations:

```cpp
#define ERASE_AT(vec, pos) vec.erase(vec.begin() + pos)
if (!vec.empty()) vec.pop_back();
if (vec.size() > 3) ERASE_AT(vec, 3);
```

### Creating a New Vector with a Default Value

The following code creates a new vector with 5 elements, all initialized to the value 7:

**Standard Version:**

```cpp
std::vector<int> vec2(5, 7);
```

- $ \text{std::vector<int> vec2(5, 7);} $ creates a vector `vec2` with 5 elements, each initialized to 7.

**Optimized for Minimal Typing:**

No significant reduction can be achieved here without compromising clarity, but using `#define` can help in repetitive situations:

```cpp
#define VI std::vector<int>
VI vec2(5, 7);
```

### Resizing and Filling with Random Values

The vector `vec2` is resized to 10 elements, and each element is filled with a random value between 1 and 100:

**Standard Version:**

```cpp
vec2.resize(10);

unsigned seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
std::mt19937 generator(seed);
std::uniform_int_distribution<int> distribution(1, 100);

for (size_t i = 0; i < vec2.size(); ++i) {
    vec2[i] = distribution(generator);
}
```

- $ \text{vec2.resize(10);} $ resizes the vector to contain 10 elements.
- The generator $ \text{std::mt19937} $ is seeded based on the current time, and the distribution generates random integers between 1 and 100.

**Optimized for Minimal Typing:**

```cpp
vec2.resize(10);
std::mt19937 gen(std::random_device{}());
std::uniform_int_distribution<int> dist(1, 100);
for (auto& v : vec2) v = dist(gen);
```

By using modern C++ constructs such as ranged-based `for` loops, we reduce the complexity of the loop and the generator initialization, making the code cleaner and more efficient to type.

### Sorting the Vector

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

## Matrices

In C++20, matrices are typically represented as vectors of vectors (`std::vector<std::vector<T>>`), where each inner vector represents a row of the matrix. This approach allows for dynamic sizing and easy manipulation of multi-dimensional data, making matrices ideal for problems involving grids, tables, or any 2D structure.

Matrices in C++ offer flexibility in managing data: you can resize rows and columns independently, access elements using intuitive indexing, and leverage standard vector operations for rows. Additionally, the use of `ranges` and `views` introduced in C++20 enhances the ability to iterate and transform matrix data more expressively and efficiently.

_The use of matrices is common in competitive programming for tasks such as implementing dynamic programming tables, graph adjacency matrices, or performing transformations on 2D data. With the powerful capabilities of C++20's STL, matrices become a highly adaptable and efficient way to handle complex, multi-dimensional computations in a structured manner_.

### Creating and Filling a Matrix

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

### Displaying the Matrix

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

### Inserting Elements at a Specific Position

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

### Removing the Last Element and a Specific Element

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

### Creating a New Vector with a Default Value

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

### Resizing and Filling with Random Values

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

### Sorting Matrices by Rows and Columns

In C++20, we can sort matrices (represented as vectors of vectors) both by rows and by columns. Here are examples of how to do both:

#### Sorting by Rows

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

### Sorting by Columns

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

### Vectors as Inputs and Outputs

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

#### Optimized Version Using `fread` and `putchar` with Command-Line File Input

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

