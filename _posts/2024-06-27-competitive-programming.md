---
layout: post
title: Competitive programming in C++ - Techniques and Insights
author: Frank
categories:
  - Matemática
  - Linguagens Formais
  - Programação
tags:
  - Matemática
  - Linguagens Formais
  - Programação Dinâmica
  - Dynamic Programming
  - Dynamic Programming
  - C++ Algorithms
  - Performance Analysis
  - Coding Examples
  - Algorithm Optimization
  - Practical Programming Guide
image: assets/images/prog_dynamic.jpeg
description: Dynamic Programming in C++ with practical examples, performance analysis, and detailed explanations to optimize your coding skills and algorithm efficiency.
slug: dynamic-programming
keywords:
  - Dynamic Programming
  - C++ Algorithms
  - Coding Examples
  - Performance Optimization
  - Algorithm Efficiency
  - Programming Guide
  - Code Comparison
  - Developer Tips
rating: 5
published: 2024-06-27T19:43:15.124Z
draft: null
featured: true
toc: true
preview: In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Dynamic Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and enhance algorithm efficiency.
beforetoc: In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Dynamic Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and enhance algorithm efficiency.
lastmod: 2024-09-09T01:43:02.316Z
---

## Introduction

Dynamic programming (DP) is one of the most essential and versatile techniques in competitive programming. It provides a structured way to solve complex problems by breaking them down into simpler subproblems and using previously computed solutions to build the final answer efficiently. By mastering dynamic programming, you equip yourself with the ability to tackle a wide range of challenges, including optimization problems, decision-making tasks, and algorithmic puzzles. DP encourages thinking in terms of state transitions, recursive relations, and memoization, which are fundamental concepts in algorithm design.

Starting with dynamic programming lays a strong foundation for competitive programming because it emphasizes critical problem-solving strategies such as identifying overlapping subproblems and recognizing optimal substructure. These strategies are crucial when working with time-constrained problems where brute-force solutions are impractical. By focusing on dynamic programming early, you gain the skills necessary to approach not only DP-specific problems but also to recognize when it can be applied in more complex, seemingly unrelated algorithmic challenges. This understanding significantly boosts your efficiency and performance in competitive programming contests.

## C++ Competitive Programming Hacks

In this section, we'll cover essential tips and tricks that will help you improve your efficiency and performance in competitive programming using C++. From mastering typing speed to reducing code verbosity and handling complexity, each aspect plays a crucial role in gaining a competitive edge.

C++ is known for its speed and flexibility, but using it effectively requires a deep understanding of both the language and the common pitfalls that arise in competitive programmings. The goal here is to help you streamline your coding process, minimize errors, and ensure that your solutions run as efficiently as possible.

We'll break down these tips into the following areas:

- **Typing Efficiency**: How to type faster and more accurately, which can save you valuable time during competitive programmings.
- **Code Reduction Techniques**: Ways to reduce code size without sacrificing clarity or correctness, using C++ features like the Standard Template Library (STL).
- **Managing Complexity**: Strategies to handle the time and space complexity of algorithms, ensuring that your solutions scale efficiently with larger inputs.

By applying these hacks, you'll be better equipped to tackle the challenges of competitive programming with C++ and improve your overall performance. Keep in mind that the code and techniques discussed here are optimized specifically for competitive programming environments, where the code is written for single-use and will not be maintained or reused. These approaches may not be suitable for professional development, where code readability, maintainability, and long-term reliability are critical.

### Typing Tips

If you don’t type quickly, **you should invest at least two hours per week** on the website: [https://www.speedcoder.net](https://www.speedcoder.net). Once you have completed the basic course, select the C++ lessons and practice regularly. Time is crucial in competitive programming, and slow typing can be disastrous.

To expand on this, efficient typing isn’t just about speed; it’s about reducing errors and maintaining a steady flow of code. When you're in a competitive programming, every second matters. Correcting frequent typos or having to look at your keyboard will significantly slow down your progress. Touch typing—knowing the layout of the keyboard and typing without looking—becomes a vital skill.

### Why Typing Speed Matters

In a typical programming competitive programming, you’ll have to solve several problems within a fixed time frame. Faster typing allows you to focus more on problem-solving rather than struggling to input the code. However, typing speed without accuracy is meaningless. **Accurate and fast typing** ensures that once you have the solution, you can implement it efficiently.

Typing slow or with many errors leads to:

- Lost time correcting mistakes
- Distractions from your problem-solving process
- Higher risk of failing to complete solutions on time

You should aim for a **typing speed** of at least 60 words per minute (WPM) with high accuracy. On platforms like [https://www.speedcoder.net](https://www.speedcoder.net), you can practice typing specific code syntax, which is more effective for programmers compared to general typing lessons. For example, learning C++ or Python shortcuts helps improve your typing speed in actual coding scenarios.

$$
\text{Time spent fixing errors} + \text{Time lost from slow typing} = \text{Lower overall performance in competitive programmings}
$$

### Advanced Typing Techniques for Programmers

Here are some additional tips to improve your typing for competitive programming:

1. **\*Use IDE shortcuts:** Learn keyboard shortcuts for your favorite Integrated Development Environment (IDE). Navigating and editing code using shortcuts reduces time spent moving between mouse and keyboard. In the case of [ICPC](https://icpc.global/) contests, the IDE provided will typically be [Eclipse](https://www.eclipse.org/downloads/packages/release/helios/sr2/eclipse-ide-cc-developers), so it’s crucial to familiarize yourself with its shortcuts and navigation to maximize efficiency during the competitive programming.
2. **Focus on frequent patterns:** As you practice, focus on typing patterns you use frequently, such as loops, if-else conditions, and function declarations. Automating these patterns in your muscle memory will save valuable time.
3. **Practice algorithm templates:** Some problems require similar algorithms, such as dynamic programming, sorting, or tree traversal. By practicing typing these algorithms regularly, you'll be able to quickly implement them during competitive programmings.

In competitive programming, every second counts, and being proficient with your typing can give you a significant advantage.

## Typing Less in Competitive Programming

In competitive programming, time is a critical resource. Therefore, optimizing typing speed and avoiding repetitive code can make a significant difference. Below, we will discuss strategies to minimize typing when working with `std::vector` during competitive programmings, where access to the internet or pre-prepared code snippets may be restricted.

### 1. Using `#define` for `std::vector` Abbreviations

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

However, it's important to note that in larger, professional projects, using `#define` for type aliases is generally discouraged because it does not respect C++ scoping rules and can lead to unexpected behavior during debugging. In competitive programming, where speed is essential, this technique can be useful, but it should be avoided in long-term or collaborative codebases.

### 2. Predefined Utility Functions

Another effective strategy is to define utility functions that you can reuse for common vector operations, such as reading from input, printing, or performing operations like sorting or summing elements.

#### Reading Vectors

```cpp
#define FAST_IO std::ios::sync_with_stdio(false); std::cin.tie(nullptr);

void read_vector(VI& vec, int n) {
    for (int i = 0; i < n; ++i) {
        int x;
        std::cin >> x;
        vec.push_back(x);
    }
}
```

With the `read_vector` function, you can quickly read a vector of `n` elements:

```cpp
VI numbers;
read_vector(numbers, n);
```

#### Printing Vectors

```cpp
void print_vector(const VI& vec) {
    for (const int& x : vec) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}
```

This function allows you to easily print the contents of a vector:

```cpp
print_vector(numbers);
```

### 3. Predefining Common Operations

If you know that certain operations, such as sorting or summing elements, are frequent in a competitive programming, consider defining these operations at the beginning of the code.

#### Sorting Vectors

```cpp
#define SORT_VECTOR(vec) std::sort(vec.begin(), vec.end())
```

You can then sort any vector quickly:

```cpp
SORT_VECTOR(numbers);
```

#### Summing Elements

```cpp
int sum_vector(const VI& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0);
}
```

To calculate the sum of a vector's elements:

```cpp
int total = sum_vector(numbers);
```

### 4. Using Lambda Functions

In C++11 and later versions, lambda functions can be a quick and concise way to define operations inline for vectors:

```cpp
auto print_square = [](const VI& vec) {
    for (int x : vec) {
        std::cout << x * x << " ";
    }
    std::cout << "\n";
};
```

These inline functions can be defined and used without the need to write complete functions:

```cpp
print_square(numbers);
```

### Prefer Not to Use `#define`

Another way to reduce typing time is by using `typedef` or `using` to create abbreviations for frequently used vector types:

```cpp
typedef std::vector<int> VI;
typedef std::vector<std::vector<int>> VVI;
using VS = std::vector<std::string>;
```

In many cases, the use of `#define` can be replaced with more modern and safe C++ constructs like `using`, `typedef`, or `constexpr`. `#define` does not respect scoping rules and does not offer type checking, which can lead to unintended behavior. Using `typedef` or `using` provides better type safety and integrates smoothly with the C++ type system, making the code more predictable and easier to debug.

#### 1. Replacing `#define` with Type Aliases

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

**`using` and `typedef` are preferred because they respect C++ scoping rules and offer better support for debugging, making the code more secure and readable**.

#### 2. Replacing `#define` with Constants

If you're using `#define` to define numeric constants, you can replace them with `constexpr`, which also integrates better with the C++ type system and offers compile-time type checking.

Example:

```cpp
#define PI 3.14159
```

Can be replaced with:

```cpp
constexpr double PI = 3.14159;
```

**`constexpr` allows the value to be treated as a constant of the correct type (`double` in this case), providing type checks and optimizing the code during compilation**.

#### 3. Using `constexpr` in Functions

If you have macros that perform calculations, you can replace them with `constexpr` functions:

Example:

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

For competitive programming, using `#define` might seem like the fastest way to reduce typing and speed up coding. However, using `typedef` or `using` is generally more efficient because it avoids potential issues with macros and integrates better with the compiler. **While reducing variable names or abbreviating functions might save time during a competitive programming, remember that in professional code, clarity and maintainability are far more important**. Therefore, avoid using shortened names and unsafe constructs like `#define` in production code, libraries, or larger projects.

## Optimizing File I/O in C++ for competitive programmings

In many competitive programming contests, especially those involving large datasets, the program is required to read input from a file that can be very large. For this reason, it is crucial to optimize how files are read. Efficient file handling can make the difference between a solution that completes within the time limits and one that does not. Implementing techniques to speed up file I/O is indispensable for handling such cases effectively.

### Disabling I/O Synchronization

To improve the performance of input/output (I/O) operations, we disable the synchronization between the standard C and C++ I/O libraries:

```cpp
std::ios::sync_with_stdio(false);
std::cin.tie(nullptr);
```

- The function `std::ios::sync_with_stdio(false)` disables the synchronization of the C++ streams with the C streams, allowing the program to perform I/O operations more quickly.
- `std::cin.tie(nullptr)` detaches the input stream (`std::cin`) from the output stream (`std::cout`), preventing automatic flushes that can cause delays.

When we disable synchronization with `std::ios::sync_with_stdio(false);`, the program benefits from better performance in I/O operations since it no longer needs to synchronize the C++ input/output functions (`std::cin`, `std::cout`) with the C functions (`scanf`, `printf`).

This synchronization, when enabled, introduces overhead because the system ensures that both libraries can be used simultaneously without conflict. By removing this synchronization, we eliminate that overhead, allowing I/O operations to be processed more directly and faster.

This optimization is particularly beneficial in programs that perform a large number of read and write operations, such as when processing large amounts of data from files. Additionally, by using `std::cin.tie(nullptr);`, we prevent `std::cout` from being automatically flushed before each input operation, avoiding another form of latency that could impact performance in I/O-heavy contexts.

### Command Line Argument Checking

Before proceeding with execution, the code checks if exactly one argument was passed through the command line, which represents the name of the file to be read:

```cpp
if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <file_name>\n";
    return 1;
}
```

- `argc` is the number of arguments passed to the program, including the name of the executable.
- `argv` is an array of strings containing the arguments.
- If the number of arguments is not 2, the program prints an error message and exits with `return 1`, indicating failure.

### Opening and Verifying the File

The code attempts to open the specified file and checks whether the opening was successful:

```cpp
std::ifstream file(argv[1]);
if (!file) {
    std::cerr << "Error opening file: " << argv[1] << "\n";
    return 1;
}
```

- `std::ifstream file(argv[1]);` attempts to open the file for reading.
- If `file` is not in a valid state (i.e., the file couldn't be opened), an error message is displayed, and the program terminates.

Of course, in competitive programmings, the input file will most often be handled by an automated testing system, so you probably won't need to check whether the file opened correctly or not.

### Introduction to File I/O in C++

In C++, file input and output (I/O) operations are handled through classes provided by the `<fstream>` library. The three main classes used for this purpose are `std::ifstream`, `std::ofstream`, and `std::fstream`. Each of these classes is specialized for different types of I/O operations.

#### `std::ifstream`: File Reading

The `std::ifstream` class (input file stream) is used exclusively for reading files. It inherits from `std::istream`, the base class for all input operations in C++.

#### Opening Files for Reading

In our code, we use `std::ifstream` to open a text file and read its contents:

```cpp
std::ifstream file(argv[1]);
```

- `std::ifstream file(argv[1]);`: Tries to open the file whose name is passed as a command-line argument. If the file cannot be opened, the `file` stream will be invalid.

Off course we can use `std::ifstream` to read files from command line:

```cpp
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    // Abre o arquivo apenas para leitura
    std::ifstream file(argv[1]);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << argv[1] << "\n";
        return 1;
    }

    std::string line;

    // Lê o conteúdo do arquivo linha por linha e exibe no console
    std::cout << "Contents of the file:\n";
    while (std::getline(file, line)) {
        std::cout << line << "\n";
    }

    file.close();
    return 0;
}
```

#### Verifying File Opening

After attempting to open the file, it’s crucial to check whether the opening was successful:

```cpp
if (!file) {
    std::cerr << "Error opening file: " << argv[1] << "\n";
    return 1;
}
```

- `if (!file)`: Checks if the `file` stream is in an invalid state (which indicates the file was not opened correctly). If the file can't be opened, an error message is displayed, and the program exits.

Again, in competitive programmings, the input file will most often be handled by an automated testing system, so you probably won't need to check whether the file opened correctly or not.

#### File Reading

Once the file is successfully opened, we can read its contents:

```cpp
std::getline(file, line);
while (file >> num) {
    vec.push_back(num);
}
```

- `std::getline(file, line);`: Reads a full line from the file and stores it in the string `line`.
- `file >> num`: Reads numbers from the file and stores them in `num`, which are then added to the vector `vec` using `vec.push_back(num);`.

#### File Closing

After finishing with a file, it should be closed to free the associated resources. This happens automatically when the `std::ifstream` object is destroyed, but it can also be done explicitly:

```cpp
file.close();
```

- `file.close();`: Closes the file manually. Although the file is automatically closed when the object goes out of scope, explicitly closing the file can be useful to ensure the data is correctly released before the program ends or before opening another file.

#### `std::ofstream`: File Writing

While we didn’t use `std::ofstream` in the provided code, it's important to mention it. The `std::ofstream` class (output file stream) is used for writing to files. It inherits from `std::ostream`, the base class for all output operations in C++.

#### Opening Files for Writing

The syntax for opening a file for writing using `std::ofstream` is similar to that of `std::ifstream`:

```cpp
std::ofstream outFile("output.txt");
```

- `std::ofstream outFile("output.txt");`: Opens or creates a file called `output.txt` for writing. If the file already exists, its contents will be truncated (erased).

#### `std::fstream`: File Reading and Writing

The `std::fstream` class combines the functionality of both `std::ifstream` and `std::ofstream`, allowing for both reading from and writing to files. It inherits from `std::iostream`, the base class for bidirectional I/O operations.

#### Opening Files for Reading and Writing

An example of how to open a file for both reading and writing would be:

```cpp
std::fstream file("data.txt", std::ios::in | std::ios::out);
```

- `std::fstream file("data.txt", std::ios::in | std::ios::out);`: Opens `data.txt` for both reading and writing. The parameter `std::ios::in | std::ios::out` specifies that the file should be opened for both input and output.

Or to use files read from command line we could use:

```cpp
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    // Abre o arquivo com leitura e escrita
    std::fstream file(argv[1], std::ios::in | std::ios::out);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << argv[1] << "\n";
        return 1;
    }

    std::string line;

    // Lê o conteúdo do arquivo e exibe no console
    std::cout << "Contents of the file:\n";
    while (std::getline(file, line)) {
        std::cout << line << "\n";
    }

    // Reposiciona o ponteiro para o início do arquivo
    file.clear(); // Limpa qualquer flag de erro
    file.seekg(0, std::ios::beg);

    // Adiciona uma nova linha ao final do arquivo
    file << "\nNew line added to the file.\n";

    // Reposiciona o ponteiro para o início novamente para leitura após a escrita
    file.clear();
    file.seekg(0, std::ios::beg);

    std::cout << "\nUpdated contents of the file:\n";

    // Lê e exibe o conteúdo atualizado do arquivo
    while (std::getline(file, line)) {
        std::cout << line << "\n";
    }

    file.close();
    return 0;
}
```

#### File Opening Modes

When opening files, we can specify different opening modes using values from the `std::ios_base::openmode` enumeration. Some of the most common modes include:

- `std::ios::in`: Open for reading (default for `std::ifstream`).
- `std::ios::out`: Open for writing (default for `std::ofstream`).
- `std::ios::app`: Open for writing at the end of the file, without truncating it.
- `std::ios::ate`: Open and move the file pointer to the end of the file.
- `std::ios::trunc`: Truncate the file (erase existing content).
- `std::ios::binary`: Open the file in binary mode.

## Advanced File I/O Techniques in C++

There are faster ways to open and process files in C++, which can be especially useful in competitive programming when dealing with large data sets. Here are some techniques that can enhance the efficiency of file handling:

1. **Disable I/O Synchronization**

   As mentioned previously, disabling the synchronization between the C and C++ I/O libraries using `std::ios::sync_with_stdio(false)` and unlinking `std::cin` from `std::cout` with `std::cin.tie(nullptr)` can significantly speed up data reading.

2. **Use Manual Buffering**

   Manual buffering allows you to read the file in large chunks, which reduces the overhead of multiple I/O operations. Below is the code, followed by a detailed explanation of how we efficiently read the entire file into a buffer:

   ```cpp
   #include <fstream>
   #include <iostream>
   #include <vector>

   int main(int argc, char* argv[]) {
       if (argc != 2) {
           std::cerr << "Usage: " << argv[0] << " <file_name>\n";
           return 1;
       }

       std::ifstream file(argv[1], std::ios::in | std::ios::binary);
       if (!file) {
           std::cerr << "Error opening file: " << argv[1] << "\n";
           return 1;
       }

       // Move file pointer to the end to determine file size
       file.seekg(0, std::ios::end);
       size_t fileSize = file.tellg();
       file.seekg(0, std::ios::beg);

       // Create buffer and read file in one go
       std::vector<char> buffer(fileSize);
       file.read(buffer.data(), fileSize);

       // Process buffer contents
       // Example: Print the first 100 characters of the file
       for (int i = 0; i < 100 && i < fileSize; ++i) {
           std::cout << buffer[i];
       }

       return 0;
   }
   ```

   Let’s break down the most important lines used to read the file efficiently:

   ```cpp
   file.seekg(0, std::ios::end);
   ```

   This line moves the file pointer to the end of the file. The `seekg` function (seek "get" position) is used to set the position of the next read operation. Here, the first argument `0` means no offset, and the second argument `std::ios::end` moves the pointer to the end of the file. This allows us to calculate the size of the file, which is essential for creating a buffer that will hold the entire file's content.

   ```cpp
   size_t fileSize = file.tellg();
   ```

   After moving the pointer to the end of the file, we use `tellg()` (tell "get" position) to retrieve the current position of the pointer, which is now at the end. Since the file pointer is at the end, the value returned by `tellg()` represents the total size of the file in bytes. This value is stored in the variable `fileSize`, which we will use to allocate a buffer large enough to hold the file’s contents.

   ```cpp
   file.seekg(0, std::ios::beg);
   ```

   Now that we know the file's size, we move the file pointer back to the beginning of the file using `seekg(0, std::ios::beg)`. The argument `std::ios::beg` means we are setting the pointer to the start of the file, where the reading will begin. This ensures we are ready to read the file from the first byte.

   ```cpp
   std::vector<char> buffer(fileSize);
   ```

   We then create a `std::vector<char>` buffer with the size `fileSize`. This buffer will store the contents of the entire file in memory. Using a `std::vector` is convenient because it automatically manages memory and provides access to the underlying data using the `data()` method.

   ```cpp
   file.read(buffer.data(), fileSize);
   ```

   Finally, we use `file.read()` to read the entire file into the buffer. The method `buffer.data()` provides a pointer to the beginning of the buffer where the file’s contents will be stored. The second argument, `fileSize`, specifies how many bytes to read from the file. Since `fileSize` equals the total size of the file, the entire file is read into the buffer in one go.

   By using `seekg()` to calculate the file size and then reading the file all at once into a buffer, we avoid multiple I/O operations, which would otherwise slow down the process. Reading the file in one operation reduces system calls and minimizes overhead, making the process faster, especially when dealing with large files.

3. **Reading Lines More Efficiently**

   Instead of using `std::getline()`, which can be relatively slow for large files, you can implement a custom buffer to store multiple lines at once, reducing the overhead of repeatedly calling the I/O functions.

### Using `mmap` for Faster File I/O in Unix-Based Systems

In competitive programming, especially in contests like ICPC where the environment is Unix-based (typically Linux), it is crucial to explore every possible optimization for handling large input files. One such technique is using the `mmap` system call, which provides an extremely fast option for reading large files by mapping them directly into memory. This allows almost instantaneous access to the file's content without multiple read operations, significantly reducing I/O overhead.

The `mmap` function maps a file or device into memory. Once the file is mapped, it behaves as if it's part of the program's memory space, allowing you to access file contents through pointer arithmetic rather than explicit file read operations. This eliminates the need for repeated system calls for reading file data, as you access the file as if it were a simple array in memory.

This approach is useful in environments like ICPC, where files can be very large, and efficiency is paramount. **However, it's important to note that `mmap` is specific to Unix-based systems and is not portable across all operating systems, such as Windows**.

#### How to Use `mmap`

Here's an example of how you can use `mmap` to read a file efficiently in C++ on a Unix-based system:

```cpp
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file_name>\n";
        return 1;
    }

    // Open the file
    int fd = open(argv[1], O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error opening file: " << argv[1] << "\n";
        return 1;
    }

    // Get the size of the file
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error getting file size\n";
        close(fd);
        return 1;
    }
    size_t fileSize = sb.st_size;

    // Memory-map the file
    char* fileData = (char*)mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (fileData == MAP_FAILED) {
        std::cerr << "Error mapping file to memory\n";
        close(fd);
        return 1;
    }

    // Process the file data (example: print the first 100 characters)
    for (size_t i = 0; i < 100 && i < fileSize; ++i) {
        std::cout << fileData[i];
    }

    // Unmap the file and close the file descriptor
    if (munmap(fileData, fileSize) == -1) {
        std::cerr << "Error unmapping file\n";
    }
    close(fd);

    return 0;
}
```

#### Explanation of Key Steps

1. **Opening the File**:

   ```cpp
   int fd = open(argv[1], O_RDONLY);
   ```

   The `open()` function opens the file specified in the command-line arguments in read-only mode. The file descriptor `fd` is returned, which is later used to map the file into memory.

2. **Getting the File Size**:

   ```cpp
   struct stat sb;
   fstat(fd, &sb);
   ```

   The `fstat()` function retrieves the size of the file and stores it in the `stat` structure. The file size is crucial for knowing how much memory to map.

3. **Mapping the File into Memory**:

   ```cpp
   char* fileData = (char*)mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
   ```

   The `mmap()` function maps the entire file into memory. The `PROT_READ` flag allows read-only access, and `MAP_PRIVATE` ensures that any modifications to the memory are private to this process (although we won't modify the file in this example). Once the file is mapped, `fileData` points to the beginning of the file's contents in memory.

4. **Processing the Data**:
   After mapping the file, you can access the file's content using `fileData` as if it were an array. For example, the above code prints the first 100 characters from the file.

5. **Unmapping and Closing**:

   ```cpp
   munmap(fileData, fileSize);
   close(fd);
   ```

   After processing the file, it is important to unmap the memory with `munmap()` and close the file descriptor with `close()`. This ensures that system resources are properly freed.

`mmap` provides several advantages when it comes to handling large file I/O. First, it offers **speed** by allowing direct access to file contents in memory, eliminating the need for repeated system calls, which significantly reduces overhead. Additionally, **simplicity** is a key benefit, as the file can be accessed like a normal array after mapping, streamlining file processing logic. Finally, **memory efficiency** is improved, as `mmap` only loads the required parts of the file into memory, avoiding the need to load the entire file into a buffer, which is especially useful for large files.

**portability**: it's important to note that the `mmap` function is specific to POSIX-compliant operating systems such as Linux, macOS, and other Unix-like systems. This function is not natively available on Windows platforms, which may limit the portability of code that uses it. For cross-platform development or in environments that include Windows systems, it's advisable to provide an alternative implementation or use libraries that offer similar functionality in a portable manner. In programming competitive programmings that occur in controlled environments, such as ICPC, where the operating system is usually specified (often Linux), the use of `mmap` may be appropriate. However, for code that needs to run on multiple platforms, consider using more universal I/O methods, such as `std::ifstream` or `fread`, which are supported on a wider range of operating systems.

### Parallel Input/Output with Threads (C++20)

CC++20 introduced several improvements for parallel programming, including the efficient use of threads and asynchronous tasks with `std::async`. In many competitive programming scenarios, input and output (I/O) operations are performed sequentially. **However, despite it being quite rare for input files to be very large in competitive programmings, in cases of intensive I/O or when there is a need to process large volumes of data simultaneously, parallel I/O can be an advantageous strategy**.

In situations with heavy I/O workloads, such as reading and processing large input files or performing intensive calculations while still reading or writing data, `std::async` and threads can be used to split operations and execute different tasks simultaneously, making the best use of available time.

### Example of Parallel I/O Using `std::async`

Below is a simple example of how to use `std::async` to perform input and output operations in parallel. In this example, while data is being read, another thread can be used to process or display the data simultaneously, optimizing the time spent on I/O operations:

```cpp
#include <iostream>
#include <future>
#include <vector>

// Function to read input from the user
void read_input(std::vector<int>& vec, int n) {
for (int i = 0; i < n; ++i) {
std::cin >> vec[i];
}
}

// Function to process and print the output
void process_output(const std::vector<int>& vec) {
for (int i : vec) {
std::cout << i << " ";
}
std::cout << std::endl;
}

int main() {
int n;
std::cin >> n;

    // Create a vector of size 'n'
    std::vector<int> numbers(n);

    // Use std::async to read the input asynchronously
    auto readTask = std::async(std::launch::async, read_input, std::ref(numbers), n);

    // Wait for the input reading to complete before proceeding
    readTask.wait();

    // Process and print the vector after reading
    process_output(numbers);

    return 0;

}
```

In this example, the `std::async` function is used to run the `read_input` function asynchronously on a separate thread. This means that the data can be read in the background while other operations are prepared or started.

`std::async` executes the `read_input` function in a new thread, passing the `numbers` vector and the number of inputs `n` as parameters. The `std::launch::async` option ensures that the function is run in parallel, and not lazily (i.e., when the result is needed). The call to `readTask.wait()` ensures that the asynchronous read operation is completed before the program continues. This synchronizes the operation, ensuring that the input is fully read before trying to process the data.

Although this example uses the main thread to process the output after reading, in more complex scenarios, both processing and reading could be parallelized, or even multiple reads could occur simultaneously, depending on the needs:

```cpp
#include <iostream>
#include <future>
#include <vector>

// Function to read input from the user
void read_input(std::vector<int>& vec, int n) {
    for (int i = 0; i < n; ++i) {
        std::cin >> vec[i];
    }
}

// Function to process and sum the input
void process_output(const std::vector<int>& vec) {
    int sum = 0;
    for (int i : vec) {
        sum += i;
    }
    std::cout << "Sum of elements: " << sum << std::endl;
}

int main() {
    int n;
    std::cin >> n;

    // Vectors to store the input data
    std::vector<int> numbers1(n);
    std::vector<int> numbers2(n);

    // Asynchronous read of the first data set
    auto readTask1 = std::async(std::launch::async, read_input, std::ref(numbers1), n);

    // Asynchronous read of the second data set in parallel
    auto readTask2 = std::async(std::launch::async, read_input, std::ref(numbers2), n);

    // Asynchronous processing of the first data set in parallel
    auto processTask = std::async(std::launch::async, [&]() {
        readTask1.wait(); // Wait for the first read to complete before processing
        process_output(numbers1); // Process the first data set
    });

    // Wait for both reads to finish
    readTask1.wait();
    readTask2.wait();

    // Output the second set of numbers
    std::cout << "Second set of numbers: ";
    for (int i : numbers2) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // Wait for the asynchronous processing to finish
    processTask.wait();

    return 0;
}
```

Using threads for parallel I/O can improve performance in scenarios where there is a large volume of data to be read or written, especially if the reading time can be _masked_ while another thread is processing data or preparing the next phase of the program.

However, this technique should be used with care. Adding threads and asynchronous operations increases code complexity, requiring careful synchronization to avoid race conditions or data inconsistencies. That's why we should avoid this technique in competitive programming. While parallelism can improve execution time, creating and managing threads also has a computational cost. In some cases, the gain may not justify the added complexity. In many competitive programming environments, I/O is simple and sequential, meaning that this technique may not always be necessary or beneficial. It should be used in scenarios with extremely heavy I/O workloads or when processing and reading/writing can be separated.

The use of parallel I/O in programming competitive programmings typically applies to scenarios where there are many read/write operations or when the program needs to process large volumes of data while still reading or writing files. This situation is usual in [AI competitive programmings](https://opencv.org/opencv-ai-competitive programming-2021/) and in hackatons. This technique can be useful in problems involving the manipulation of large datasets or intensive input/output processing, such as in "big data" challenges or reading/writing from disks. However, due to its complexity, the use of `std::async` and threads should be reserved for cases where parallelism offers a significant advantage over traditional sequential I/O.

### Maximizing Input/Output Efficiency in Competitive Programming (Windows and Linux)

In some competitive programming environments, inputs are provided via the command line. The first input is the size of the array, followed by the array elements separated by spaces. Efficiently reading this data and outputting the result is crucial, especially when dealing with large datasets. Below is an approach to handle input and output in the fastest way for both Windows and Linux.

#### Optimized Input and Output

The following example demonstrates how to read inputs and output results efficiently in C++ using the fastest I/O methods available on both Windows and Linux.

```cpp
#include <iostream>
#include <vector>
#include <cstdio>

int main() {
    // Disable synchronization for faster I/O
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Read the size of the array
    int n;
    std::cin >> n;

    // Create a vector to store the array elements
    std::vector<int> arr(n);

    // Read the elements of the array
    for (int i = 0; i < n; ++i) {
        std::cin >> arr[i];
    }

    // Output the array elements
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
    return 0;
}
```

### Key Techniques for Faster I/O

1. **Disabling I/O Synchronization**:
   The line `std::ios::sync_with_stdio(false);` disables the synchronization between the C and C++ I/O streams. This allows the program to perform I/O operations faster because it no longer needs to synchronize `std::cin` and `std::cout` with `scanf` and `printf`.

2. **Unlinking `cin` and `cout`**:
   The line `std::cin.tie(nullptr);` ensures that `std::cout` will not be flushed automatically before every `std::cin` operation, which can slow down the program. By unlinking them, you have more control over when output is flushed.

### Differences Between Windows and Linux

On both **Windows** and **Linux**, the above code will work efficiently. However, since competitive programming platforms often use Linux, the synchronization of I/O streams plays a more significant role in Linux environments. Disabling synchronization is more crucial on Linux for achieving maximum performance, while the effect may be less noticeable on Windows. Nevertheless, the method remains valid and provides optimal speed in both environments.

### Input and Output Through Standard Methods

While `std::cin` and `std::cout` are often fast enough after synchronization is disabled, some competitive programmings on Unix-based systems like ICPC allow even faster input methods using `scanf` and `printf`. Below is an alternative version that uses `scanf` and `printf` for faster input/output handling:

```cpp
#include <cstdio>
#include <vector>

int main() {
    // Read the size of the array
    int n;
    scanf("%d", &n);

    // Create a vector to store the array elements
    std::vector<int> arr(n);

    // Read the elements of the array
    for (int i = 0; i < n; ++i) {
        scanf("%d", &arr[i]);
    }

    // Output the array elements
    for (int i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }

    printf("\n");
    return 0;
}
```

It is important to highlight that `scanf` and `printf` are widely recognized as insecure functions due to their lack of built-in protections against common vulnerabilities such as buffer overflows. We are discussing them here only because the code created for competitive programming is typically used only once during a contest, and the primary focus is on speed and efficiency. However, these functions — and any others considered unsafe (see [stackoverflow](https://stackoverflow.com/questions/2565727/which-functions-from-the-standard-library-must-should-be-avoided))— should never be used in production code, libraries, or any other software outside the competitive programming environment. In professional development, you should always prefer safer alternatives such as `std::cin` and `std::cout`, which provide better type safety and avoid common vulnerabilities associated with older C-style I/O functions.

### Usar Buffers Manuais com `fread` e `fwrite`

Embora funções como `scanf` e `printf` sejam rápidas, o uso de `fread` e `fwrite` permite ler e escrever dados em grandes blocos, reduzindo o número de chamadas de sistema para I/O. Isso é particularmente útil ao lidar com grandes volumes de dados, pois a sobrecarga de múltiplas operações de leitura e escrita pode ser significativa.

A função `fread` é usada para ler um número especificado de bytes de um arquivo ou do `stdin` (entrada padrão) e armazena esses dados em um buffer que você define. Ao fazer uma única leitura de um grande bloco de dados, você minimiza as chamadas ao sistema operacional, o que reduz o overhead e aumenta a eficiência.

Exemplo de leitura com `fread`:

```cpp
#include <cstdio>
#include <vector>

int main() {
    char buffer[1024];  // Buffer manual de 1 KB
    size_t bytesRead = fread(buffer, 1, sizeof(buffer), stdin);

    // Processar os dados lidos
    for (size_t i = 0; i < bytesRead; ++i) {
        // Usar putchar para imprimir os dados do buffer
        putchar(buffer[i]);
    }

    return 0;
}
```

A função `fread` lê até um número especificado de itens de um fluxo de dados e os armazena no buffer fornecido. No exemplo acima, `fread(buffer, 1, sizeof(buffer), stdin)` lê até 1024 bytes da entrada padrão (`stdin`) e armazena esses dados no buffer. O número de bytes lidos é retornado como `bytesRead`.

A função `putchar` imprime um caractere de cada vez no `stdout` (saída padrão). No exemplo, usamos `putchar(buffer[i])` para imprimir cada caractere armazenado no buffer. Essa função é eficiente para lidar com dados de baixo nível, especialmente em situações em que você está processando caracteres individuais.

Comparando com `scanf` e `printf`, que são mais convenientes quando se precisa de formatação específica, como ler números inteiros ou strings, `fread` e `fwrite` são mais eficientes para grandes volumes de dados "brutos" sem formatação, como arquivos binários ou grandes blocos de texto.

Se precisar gravar dados de maneira igualmente eficiente, pode-se usar `fwrite` para escrever blocos de dados em um arquivo ou no `stdout`.

Exemplo de escrita com `fwrite`:

```cpp
#include <cstdio>
#include <vector>

int main() {
    const char* data = "Outputting large blocks of data quickly\n";
    size_t dataSize = strlen(data);

    // Escreve o buffer de dados para o stdout
    fwrite(data, 1, dataSize, stdout);

    return 0;
}
```

A função `fwrite` funciona de maneira semelhante a `fread`, mas em vez de ler dados, ela grava o conteúdo de um buffer em um arquivo ou na saída padrão. No exemplo acima, `fwrite(data, 1, dataSize, stdout)` grava `dataSize` bytes do buffer `data` no `stdout`.

Usar buffers manuais com `fread` e `fwrite` pode melhorar significativamente a performance em competições ao reduzir o número de chamadas ao sistema, o que é particularmente útil quando se lida com grandes volumes de dados. Essa técnica oferece maior controle sobre o processo de I/O e permite otimizações em situações de alto desempenho. No entanto, quando você precisa de formatação avançada, `scanf` e `printf` ainda podem ser mais convenientes e adequados.

## Introduction to Namespaces

In C++, **namespaces** are used to organize code and prevent name conflicts, especially in large projects or when multiple libraries are being used that may have functions, classes, or variables with the same name. They provide a scope for identifiers, allowing developers to define functions, classes, and variables without worrying about name collisions.

A **namespace** is a declarative region that provides a scope to the identifiers (names of types, functions, variables, etc.) inside it. This allows different parts of a program or different libraries to have elements with the same name without causing ambiguity.

### Basic Syntax of a Namespace

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

The `MyNamespace` namespace encapsulates `myFunction` and `MyClass`, preventing these names from conflicting with others of the same name in different namespaces.

#### Using Namespaces

To access elements inside a namespace, you can use the **scope resolution operator** `::`.

The **scope resolution operator** (`::`) in C++ is used to define or access elements that are within a specific scope, such as namespaces or class members. It allows the programmer to disambiguate between variables, functions, or classes that might have the same name but are defined in different contexts. For example, if a function is defined in a specific namespace, the scope resolution operator is used to call that function from the correct namespace. Similarly, within a class, it can be used to define a function outside the class declaration or to refer to static members of the class.

In competitive programming, the scope resolution operator is often used to access elements from the `std` namespace, such as `std::cout` or `std::vector`. This ensures that the standard library components are used correctly without introducing ambiguity with any other variables or functions that might exist in the global scope or within other namespaces. **Although not as common in short competitive programming code, the operator becomes critical in larger projects to maintain clear and distinct references to elements that may share names across different parts of the program**.

#### Accessing Elements of a Namespace

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

#### `using namespace std;`

The **std** namespace is the default namespace of the C++ Standard Library. It contains all the features of the standard library, such as `std::vector`, `std::cout`, `std::string`, and more.

The statement `using namespace std;` allows you to use all elements of the `std` namespace without needing to prefix them with `std::`. This can make the code more concise and readable, especially in small programs or educational examples. Additionally, it reduces typing, which is beneficial when time is limited and valuable, such as during competitive programmings.

#### Example Without `using namespace std;`

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

#### Example With `using namespace std;`

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

### Disadvantages of Using `using namespace std;`

While using `using namespace std;` makes your code shorter and easier to read, it comes with some drawbacks. In larger projects or when working with multiple libraries, it increases the likelihood of name conflicts, where different namespaces contain elements with the same name. It can also lead to ambiguity, making it less clear where certain elements are coming from, which complicates code maintenance and comprehension. Because of these risks, using `using namespace std;` is generally discouraged in production code, especially in large projects or collaborative settings.

### Alternatives to `using namespace std;`

To avoid the risks associated with `using namespace std;`, one option is to import specific elements from the `std` namespace. For example, instead of importing the entire namespace, you can import only the functions and types you need, such as `std::cout` and `std::vector`. This approach reduces the risk of name conflicts while still allowing for more concise code.

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

Another option is to keep using the `std::` prefix throughout your code. Although it requires more typing, this approach makes it clear where each element comes from and completely avoids name conflicts.

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

To maintain clean, maintainable code, it's recommended to avoid `using namespace std;` in header files, as this can force all files that include the header to import the `std` namespace, increasing the risk of conflicts. If you decide to use `using`, it is better to do so in a limited scope, such as inside a function, to minimize its impact. Adopting a consistent approach to namespaces throughout your project also improves readability and makes collaboration easier.

#### Advanced Example: Nested Namespace

Namespaces can also be nested to better organize the code.

```cpp
namespace Company {
    namespace Project {
        class ProjectClass {
        public:
            void projectMethod();
        };
    }
}

int main() {
    Company::Project::ProjectClass obj;
    obj.projectMethod();
    return 0;
}
```

Nested namespaces allow for a more hierarchical organization of code, which is particularly useful in large projects with multiple modules. However, it can make accessing elements more complex, as the full namespace hierarchy must be used.

**In competitive programming, it is generally unnecessary and inefficient to create or use custom namespaces beyond the standard `std` namespace. Since competitive programming code is typically small and written for single-use, the overhead of managing custom namespaces adds complexity without providing significant benefits. Additionally, custom namespaces are designed to prevent name conflicts in large projects with multiple libraries, but in competitive environments where the focus is on speed and simplicity, such conflicts are rare. Therefore, it is best to avoid using namespaces beyond `std` in competitive programming, and reserve namespace management for larger codebases with extensive dependencies and libraries.**

## Working with Vectors

Vectors are one of the most versatile data structures used in competitive programming. Many problems require frequent operations such as inserting, removing, resizing, and even manipulating entire matrices (2D vectors). Below we review common operations on vectors, showcasing standard implementations followed by optimized versions to minimize typing, which can be beneficial in time-critical competitive programmings.

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

- $ \text{constexpr void sort_vector(std::vector<int}\& vec);} $ is a type-safe way to define a reusable sorting function.
- This method avoids the pitfalls of `#define`, such as lack of scoping and type checking, while still minimizing the amount of typing.

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

- $ \text{std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));} $ creates a matrix of size 2x2.
- The nested `for` loop fills each element of the matrix with 1.

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

To summarize, reducing typing in competitive programming can help you focus more on problem-solving than on code verbosity. While using macros like `#define` can be helpful in short contests, be cautious of overusing them in professional code. Instead, consider using more modern and safe C++ techniques, such as `using` and `typedef`, which respect scoping rules and improve the reliability of your code.

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

### Optimized Version Using `fread` and `putchar` with Command-Line File Input

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

## Using `std::span` for Efficient Data Handling (C++20)

The `std::span` is a new feature introduced in C++20 that allows you to create lightweight, non-owning views of arrays and containers, such as `std::vector`. This avoids unnecessary copying of data and provides a flexible and efficient way to access and manipulate large blocks of data. `std::span` can be particularly useful when working with large datasets, file I/O, or when optimizing memory usage in competitive programming.

Unlike containers such as `std::vector`, `std::span` doesn't own the data it references. This means it doesn't allocate new memory and works directly with existing data, leading to lower memory overhead. Additionally, `std::span` can work with both static arrays and dynamic containers (like `std::vector`) without requiring copies. It provides safer array handling compared to raw pointers, as it encapsulates size information. Since `std::span` eliminates the need for memory copies, it can speed up operations where large datasets need to be processed in-place, or only certain views of data are required.

### Example of `std::span` for Efficient Data Access

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

### How `std::span` Works:

$ \text{std::span<int> view(numbers);} $ creates a non-owning view of the `std::vector<int>` `numbers`. This allows access to the elements of the vector without copying them. The loop $ \text{for (int num : view)} $ iterates over the elements in the `std::span`, just like it would with the original `std::vector`, but with no additional overhead from copying the data.

### Efficient Use Cases for `std::span`

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

In this example, the function `process_data` accepts a `std::span`, avoiding unnecessary copies and keeping the original data structure intact.

### Comparing `std::span` to Traditional Methods

| Feature          | `std::vector`           | Raw Pointers          | `std::span`     |
| ---------------- | ----------------------- | --------------------- | --------------- |
| Memory Ownership | Yes                     | No                    | No              |
| Memory Overhead  | High (allocates memory) | Low                   | Low             |
| Bounds Safety    | High                    | Low                   | High            |
| Compatibility    | Works with STL          | Works with raw arrays | Works with both |

Unlike `std::vector`, which manages its own memory, `std::span` does not allocate or own memory. This is similar to raw pointers but with added safety since `std::span` knows its size. `std::span` is safer than raw pointers because it carries bounds information, helping avoid out-of-bounds errors. While raw pointers offer flexibility, they lack the safety features provided by modern C++.

### Practical Application: Using `std::span` in Competitive Programming

When working with large datasets in competitive programming, using `std::span` avoids unnecessary memory copies, making operations faster and more efficient. You can easily pass sub-ranges of data to functions without creating temporary vectors or arrays. Additionally, it allows you to maintain full control over memory without introducing complex ownership semantics, as with `std::unique_ptr` or `std::shared_ptr`.

**\* Example: Efficiently Passing Data in a Competitive Programming Scenario**

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

## Efficient Data Manipulation with `std::ranges` in C++20

C++20 introduced the `<ranges>` library, which brings a powerful and flexible way to work with sequences of data through lazy-evaluated views and composable transformations. `std::ranges` allows you to create views over containers or arrays without modifying them or creating unnecessary copies. This is especially beneficial in competitive programming and high-performance applications, where minimizing both memory and computational overhead is crucial.

In traditional programming with containers like `std::vector`, iterating over and transforming data often requires intermediate storage or manual loops to handle operations like filtering, transforming, or slicing the data. With `std::ranges`, these operations can be composed in a clean and expressive way while maintaining optimal performance through lazy evaluation. Lazy evaluation means that the transformations are only computed when the data is accessed, rather than immediately creating new containers or applying operations.

### How `std::ranges` Works

The core idea behind `std::ranges` is to create "views" over data. These views allow you to manipulate and query data without modifying the underlying container. A view in `std::ranges` is an abstraction that can represent any sequence of elements that can be iterated over, just like a container. The key difference is that a view is not required to own its elements; instead, it provides a "window" into an existing data sequence, allowing for efficient operations.

#### Example: Filtering and Transforming Data with `std::ranges`

Suppose we have a vector of integers and we want to filter out the odd numbers and then multiply the remaining even numbers by two. Using traditional methods, we would need to loop through the vector, apply conditions, and store the results in a new container. With `std::ranges`, this can be done in a more expressive and efficient way:

```cpp
#include <iostream>
#include <vector>
#include <ranges>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // Create a lazy-evaluated view that filters out odd numbers and doubles the even ones
    auto even_doubled = numbers
                        | std::ranges::views::filter([](int n) { return n % 2 == 0; })
                        | std::ranges::views::transform([](int n) { return n * 2; });

    // Iterate over the view and print the results
    for (int num : even_doubled) {
        std::cout << num << " ";  // Output: 4 8 (only even numbers doubled)
    }
    std::cout << std::endl;

    return 0;
}
```

In this example, we create a view `even_doubled` over the original vector `numbers`. The first operation, `std::ranges::views::filter`, filters out all the odd numbers from the vector. The second operation, `std::ranges::views::transform`, multiplies each of the remaining even numbers by two. Both of these operations are lazily evaluated, meaning that no new container is created, and the transformations are applied only when iterating over the view. This approach is not only cleaner in terms of code but also more efficient in terms of performance.

### Composition of Operations

One of the key strengths of `std::ranges` is its composability. Operations like filtering, transforming, or slicing can be composed together, and the result is still a view. This means that you can chain multiple operations together without needing intermediate containers or data structures. The result is a highly efficient pipeline of operations that is applied only when the data is accessed.

Consider the following example, where we filter, transform, and take only a part of the data:

```cpp
#include <iostream>
#include <vector>
#include <ranges>

int main() {
    std::vector<int> numbers = {10, 15, 20, 25, 30, 35, 40};

    // Filter out numbers less than 20, double the remaining, and take only the first three
    auto result = numbers
                  | std::ranges::views::filter([](int n) { return n >= 20; })
                  | std::ranges::views::transform([](int n) { return n * 2; })
                  | std::ranges::views::take(3);

    // Iterate over the view and print the results
    for (int num : result) {
        std::cout << num << " ";  // Output: 40 50 60
    }
    std::cout << std::endl;

    return 0;
}
```

In this example, we chain together three operations: filtering the numbers greater than or equal to 20, doubling them, and taking only the first three results. The operations are applied lazily and are only computed when iterating over the final view, `result`. This leads to highly efficient data processing, as no intermediate containers are created, and each transformation is performed only once for the relevant elements.

### Memory and Performance Considerations

The key advantage of `std::ranges` is its use of lazy evaluation, which minimizes memory usage by avoiding the creation of temporary containers. In traditional methods, each operation (e.g., filtering or transforming) might create a new container, leading to increased memory consumption and computational overhead. With `std::ranges`, the operations are "stacked" and evaluated only when needed. This reduces the memory footprint and ensures that performance remains high, even when dealing with large datasets.

Another performance benefit comes from the fact that `std::ranges` operations are highly optimized. Since the operations are evaluated lazily and directly on the data, there's no need for unnecessary copying or allocation. This leads to more efficient cache usage and fewer CPU cycles spent on managing intermediate data structures.

### Practical Use Cases in Competitive Programming

In competitive programming, where speed and efficiency are critical, `std::ranges` can be a valuable tool for handling large datasets, filtering inputs, or applying transformations without introducing overhead. For example, when dealing with input that needs to be filtered, transformed, or sliced before solving a problem, using `std::ranges` can save both time and memory.

Imagine a scenario where you need to process only a portion of the input data based on certain conditions. Using traditional methods, this might involve creating multiple containers or applying multiple iterations over the data. With `std::ranges`, you can chain these operations in a single pass, improving both performance and code readability.

Consider the following example in a competitive programming context:

```cpp
#include <iostream>
#include <vector>
#include <ranges>
#include <algorithm>

int main() {
    std::vector<int> data = {50, 40, 30, 20, 10, 5};

    // Sort the data, filter values greater than 15, and transform them by subtracting 5
    auto processed = data
                     | std::ranges::views::sort
                     | std::ranges::views::filter([](int n) { return n > 15; })
                     | std::ranges::views::transform([](int n) { return n - 5; });

    // Iterate and output the results
    for (int num : processed) {
        std::cout << num << " ";  // Output: 15 25 35 45
    }
    std::cout << std::endl;

    return 0;
}
```

Here, the data is sorted, filtered, and transformed in a single efficient chain of operations. Each step is evaluated lazily, meaning that no intermediate containers or data copies are made, and each number is processed only once.

`std::ranges` in C++20 brings a powerful new way to work with data by providing efficient, lazy-evaluated views over containers. This minimizes memory usage, avoids unnecessary copying, and allows for highly optimized data processing pipelines. In competitive programming and high-performance applications, where every CPU cycle and byte of memory counts, using `std::ranges` can significantly improve both performance and code clarity. Whether you're filtering, transforming, or composing operations, `std::ranges` allows you to build complex data processing pipelines that are both expressive and efficient.

## Efficient Techniques for File I/O and Array Handling in Competitive Programming

| Function/Operation               | Most Efficient Technique                                                                        | Description                                                                                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reading from file (command line) | `std::ifstream` or `fread`/`mmap`                                                               | `std::ifstream` is efficient for small file reads, but `fread` and `mmap` are preferred for large files as they reduce system call overhead.         |
| Reading from standard input      | Disable synchronization with `std::ios::sync_with_stdio(false)` and use `std::cin.tie(nullptr)` | Disables C/C++ stream synchronization to improve performance when reading from `std::cin`.                                                           |
| Writing to terminal              | `putchar` or `printf`                                                                           | `putchar` is most efficient for writing individual characters, while `printf` is faster than `std::cout` in competitive programming.                 |
| Working with arrays              | `std::vector` with `std::span` (C++20)                                                          | `std::span` allows access to arrays and vectors without additional copies, providing bounds safety and efficiency in data handling without overhead. |
| Data processing                  | `std::ranges` (C++20)                                                                           | `std::ranges` enables efficient, lazy-evaluated chained operations like filtering and transforming data without extra memory allocation.             |
| Parallel I/O                     | `std::async` with asynchronous read and write operations                                        | `std::async` improves performance in high I/O scenarios by enabling parallel read/write operations.                                                  |
| Vector manipulation              | `std::vector` with preprocessing (e.g., macros, `constexpr`)                                    | Using macros or `constexpr` for frequent operations like sorting or summing elements can save time in competitive programmings.                      |
| Handling large data volumes      | Manual buffering with `fread` and `fwrite`                                                      | `fread` and `fwrite` allow efficient reading and writing of large blocks of data, minimizing system call overhead.                                   |

## Time and Space Complexity in Competitive Programming

In this section, we will delve deeper into understanding both time and space complexities, providing a more comprehensive look into how these affect the efficiency of algorithms, particularly in competitive programming environments. This includes examining loops, recursive algorithms, and how various complexity classes dictate algorithm performance. We'll also consider the impact of space complexity and memory usage, which is crucial when dealing with large datasets.

### Loops and Time Complexity

**One of the most common reasons for slow algorithms is the presence of multiple loops iterating over input data**. The more nested loops an algorithm contains, the slower it becomes. If there are $k$ nested loops, the time complexity becomes $O(n^k)$.

For instance, the time complexity of the following code is $O(n)$:

```cpp
for (int i = 1; i <= n; i++) {
    // code
}
```

And the time complexity of the following code is $O(n^2)$ due to the nested loops:

```cpp
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
        // code
    }
}
```

### Space Complexity in Loops

While the focus is often on time complexity, it's equally important to consider space complexity, especially when handling large inputs. A loop like the one below has a time complexity of $O(n)$ but also incurs a space complexity of $O(n)$ if an array is created to store values:

```cpp
std::vector<int> arr(n);
for (int i = 1; i <= n; i++) {
    arr.push_back(i);
}
```

**In competitive programming, excessive memory use can cause the program to exceed memory limits**. Therefore, always account for the space complexity of your solution, particularly when using arrays, matrices, or data structures that grow with input size.

## Order of Growth

Time complexity doesn't tell us the exact number of times the code within a loop executes but rather gives the order of growth. In the following examples, the code inside the loop executes $3n$, $n+5$, and $\lfloor n/2 \rfloor$ times, but the time complexity of each code is still $O(n)$:

```cpp
for (int i = 1; i <= 3*n; i++) {
    // code
}
```

```cpp
for (int i = 1; i <= n+5; i++) {
    // code
}
```

```cpp
for (int i = 1; i <= n; i += 2) {
    // code
}
```

Another example where time complexity is $O(n^2)$:

```cpp
for (int i = 1; i <= n; i++) {
    for (int j = i+1; j <= n; j++) {
        // code
    }
}
```

### Phases and Time Complexity

When an algorithm consists of consecutive phases, the total time complexity is the largest time complexity of any single phase. This is because the slowest phase typically becomes the bottleneck of the code.

For instance, the following code has three phases with time complexities of $O(n)$, $O(n^2)$, and $O(n)$, respectively. Thus, the total time complexity is $O(n^2)$:

```cpp
for (int i = 1; i <= n; i++) {
    // phase 1 code
}
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
        // phase 2 code
    }
}
for (int i = 1; i <= n; i++) {
    // phase 3 code
}
```

#### Space Complexity of Multiple Phases

When analyzing algorithms that consist of multiple phases, consider that each phase may also introduce additional memory usage. In the example above, if phase 2 allocates a matrix of size $n \times n$, the space complexity would increase to $O(n^2)$, matching the time complexity.

### Multiple Variables and Time Complexity

Sometimes, time complexity depends on multiple factors. In this case, the formula for time complexity includes multiple variables. For example, the time complexity of the following code is $O(nm)$:

```cpp
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
        // code
    }
}
```

#### Space Complexity with Multiple Variables

If the above algorithm also uses a data structure such as a matrix of size $n \times m$, the space complexity would also be $O(nm)$, increasing memory usage significantly, particularly for large input sizes.

### Recursive Algorithms

The time complexity of a recursive function depends on the number of times the function is called and the time complexity of a single call. The total time complexity is the product of these values.

For example, consider the following function:

```cpp
void f(int n) {
    if (n == 1) return;
    f(n-1);
}
```

The call `f(n)` makes $n$ recursive calls, and the time complexity of each call is $O(1)$. Thus, the total time complexity is $O(n)$.

#### Exponential Recursion

Consider the following function, which makes two recursive calls for every input:

```cpp
void g(int n) {
    if (n == 1) return;
    g(n-1);
    g(n-1);
}
```

Here, each function call generates two other calls, except when $n = 1$. The table below shows the function calls for a single initial call to $g(n)$:

| Function Call | Number of Calls |
| ------------- | --------------- |
| $g(n)$        | 1               |
| $g(n-1)$      | 2               |
| $g(n-2)$      | 4               |
| ...           | ...             |
| $g(1)$        | $2^{n-1}$       |

Thus, the time complexity is:

$$
1 + 2 + 4 + \cdots + 2^{n-1} = 2^n - 1 = O(2^n)
$$

#### Space Complexity in Recursive Functions

Recursive functions also have space complexity considerations. Each recursive call adds to the call stack, and in the case of deep recursion (like in the exponential example above), this can lead to $O(n)$ space complexity. Be cautious with recursive algorithms, as exceeding the maximum stack size can cause a program to fail due to stack overflow.

### Common Complexity Classes

Here is a list of common time complexities of algorithms:

- $O(1)$: A constant-time algorithm doesn't depend on the input size. A typical example is a direct formula calculation.
- $O(\log n)$: A logarithmic algorithm often halves the input size at each step, such as binary search.

- $O(\sqrt{n})$: Slower than $O(\log n)$ but faster than $O(n)$, this complexity might appear in algorithms that involve square root reductions in input size.

- $O(n)$: A linear-time algorithm processes the input a constant number of times.

- $O(n \log n)$: Common in efficient sorting algorithms (e.g., mergesort, heapsort), or algorithms using data structures with $O(\log n)$ operations.

- $O(n^2)$: Quadratic complexity, often seen with nested loops processing all pairs of input elements.

- $O(n^3)$: Cubic complexity arises with three nested loops, such as algorithms processing all triples of input elements.

- $O(2^n)$: This complexity usually indicates exponential growth, common in recursive algorithms that explore all subsets.

- $O(n!)$: Common in algorithms that generate all permutations of the input.

### Estimating Efficiency

When calculating an algorithm's time complexity, you can estimate whether it will be efficient enough for the given problem before implementation. A modern computer can perform hundreds of millions of operations per second.

For example, assume that the input size is $n = 10^5$. If the time complexity is $O(n^2)$, the algorithm would perform roughly $(10^5)^2 = 10^{10}$ operations, which would take several seconds, likely exceeding the time limits of most competitive programming environments.

On the other hand, given the input size, we can estimate the required time complexity of an algorithm. The following table provides useful estimates, assuming a time limit of one second:

| Input Size    | Required Time Complexity |
| ------------- | ------------------------ |
| $n \leq 10$   | $O(n!)$                  |
| $n \leq 20$   | $O(2^n)$                 |
| $n \leq 500$  | $O(n^3)$                 |
| $n \leq 5000$ | $O(n^2)$                 |
| $n \leq 10^6$ | $O(n \log n)$ or $O(n)$  |
| $n$ is large  | $O(1)$ or $O(\log n)$    |

For example, if the input size is $n = 10^5$, it is likely that the algorithm must have a time complexity of $O(n)$ or $O(n \log n)$. This insight can help guide the design of the algorithm and eliminate approaches that would result in worse time complexity.

While time complexity is a good estimate of efficiency, it hides constant factors. For example, an $O(n)$ algorithm might perform $n/2$ or $5n$ operations, and these constants can significantly affect the actual running time.

Since loops have a significant impact on code performance, we can dive deeper into the possible loop options available.

## Deep Dive into `for` Loops in Competitive Programming

C++ provides several ways to iterate over elements in a vector, using different types of `for` loops. In this section, we will explore the various `for` loop options available in C++20, discussing their performance and code-writing efficiency. We will also analyze which loops are best suited for competitive programming based on input size—whether dealing with small or large datasets.

### 1. `for` Loop with Iterator

The `for` loop using iterators is one of the most efficient ways to iterate over a vector, especially for complex operations where you need to manipulate the elements or the iterator’s position directly.

```cpp
for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";
}
```

Utilizing iterators directly avoids unnecessary function calls such as `operator[]` and allows fine-grained control over the iteration. Ideal when detailed control over the iterator is necessary or when iterating over containers that do not support direct index access (e.g., `std::list`).

**Input Size Consideration**:

- **For Small Inputs**: This is a solid option as it allows precise control over the iteration with negligible overhead.
- **For Large Inputs**: Highly efficient due to minimal overhead and memory usage. However, ensure that the iterator’s operations do not induce cache misses, which can slow down performance for large datasets.

### 2. Classic `for` Loop with Index

The classic `for` loop using an index is efficient and provides precise control over the iteration process.

```cpp
for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << vec[i] << " ";
}
```

Accessing elements via index is fast, but re-evaluating `vec.size()` in each iteration can introduce a small overhead. Useful when you need to access or modify elements by their index or when you may need to adjust the index inside the loop.

**Input Size Consideration**:

- **For Small Inputs**: Efficient and straightforward, especially when the overhead of re-evaluating `vec.size()` is negligible.
- **For Large Inputs**: If performance is critical, store `vec.size()` in a separate variable before the loop to avoid repeated function calls, which can become significant for larger datasets.

### 3. Range-Based `for-each` with Constant Reference

Range-based `for-each` with constant reference is highly efficient for reading elements since it avoids unnecessary copies.

```cpp
for (const auto& elem : vec) {
    std::cout << elem << " ";
}
```

Using constant references avoids copying, making it very efficient for both memory and execution time. Recommended for reading elements when you don’t need to modify values or access their indices.

**Input Size Consideration**:

- **For Small Inputs**: Ideal for minimal syntax and efficient execution.
- **For Large Inputs**: Excellent choice due to the avoidance of element copies, ensuring optimal memory usage and performance.

### 4. Range-Based `for-each` by Value

The `for-each` loop can also iterate over elements by value, which is useful when you want to work with copies of the elements.

```cpp
for (auto elem : vec) {
    std::cout << elem << " ";
}
```

Elements are copied, which can reduce performance, especially for large data types. Useful when you need to modify a copy of the elements without affecting the original vector.

**Input Size Consideration**:

- **For Small Inputs**: Suitable when the overhead of copying is negligible, especially if you need to modify copies of elements.
- **For Large Inputs**: Avoid for large datasets or large element types, as the copying can lead to significant performance degradation.

### 5. `for` Loop with Range Views (C++20)

C++20 introduced `range views`, which allow iteration over subsets or transformations of elements in a container without creating copies.

```cpp
for (auto elem : vec | std::views::reverse) {
    std::cout << elem << " ";
}
```

Range views allow high-performance operations, processing only the necessary elements. Ideal for operations involving transformations, filtering, or iterating over subsets of elements.

**Input Size Consideration**:

- **For Small Inputs**: Works well, especially when applying transformations like reversing or filtering, while maintaining code readability.
- **For Large Inputs**: Very efficient as no extra memory is allocated, and the processing is done lazily, meaning only the required elements are accessed.

### 6. Parallel `for` Loop (C++17/C++20)

While not a traditional `for` loop, using parallelism in loops is a powerful feature introduced in C++17 and further enhanced in C++20.

```cpp
#include <execution>

std::for_each(std::execution::par, vec.begin(), vec.end(), [](int& elem) {
elem \*= 2; // Parallelized operation
});
```

Uses multiple threads to process elements in parallel, offering substantial performance gains for intensive operations that can be performed independently on large datasets. It requires more setup and understanding of parallelism concepts but can provide significant performance boosts for operations on large datasets.

**Input Size Consideration**:

- **For Small Inputs**: Overkill. The overhead of managing threads and synchronization outweighs the benefits for small datasets.
- **For Large Inputs**: Extremely efficient. When dealing with large datasets, parallel processing can drastically reduce runtime, especially for computationally expensive operations.

### Optimal `for` Loops for Competitive Programming

Choosing the right type of `for` loop in competitive programming depends largely on input size and the specific use case. The following table summarizes the best choices for different scenarios:

| Input Size      | Best `for` Loop Option                                             | Reasoning                                                                                            |
| --------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Small           | Range-Based `for-each` with Constant Reference                     | Offers minimal syntax, high readability, and avoids copies, making it fast and efficient.            |
| Small           | Classic `for` Loop with Index                                      | Provides precise control over the index, useful when index manipulation or modification is required. |
| Large           | Iterator-Based `for` Loop                                          | Highly efficient for large datasets due to minimal memory overhead and optimized performance.        |
| Large           | Parallel `for` Loop with `std::for_each` and `std::execution::par` | Ideal for computationally heavy tasks on large datasets, leveraging multiple threads to parallelize. |
| Transformations | `for` Loop with Range Views (C++20)                                | Ideal for processing subsets or transformations of data without creating extra copies.               |

## Now the `while` Loop which we all love

The `while` loop is another fundamental control structure in C++ that is often used in competitive programming. It repeatedly executes a block of code as long as a specified condition evaluates to true. In this section, we will explore the different use cases for `while` loops, their performance considerations, and scenarios where they may be preferable to `for` loops. We will also examine their application with both small and large datasets.

### 1. Basic `while` Loop

A `while` loop continues executing its block of code until the condition becomes false. This makes it ideal for situations where the number of iterations is not known beforehand.

```cpp
int i = 0;
while (i < n) {
    std::cout << i << " ";
    i++;
}
```

The `while` loop is simple and provides clear control over the loop's exit condition. The loop runs while `i < n`, and the iterator `i` is incremented manually within the loop. This offers flexibility in determining when and how the loop terminates.

**Input Size Consideration**:

- **For Small Inputs**: This structure is efficient, especially when the number of iterations is small and predictable.
- **For Large Inputs**: The `while` loop can be optimized for larger inputs by ensuring that the condition is simple to evaluate and that the incrementing logic doesn't introduce overhead.

### 2. `while` Loop with Complex Conditions

`while` loops are particularly useful when the condition for continuing the loop involves complex logic that cannot be easily expressed in a `for` loop.

```cpp
int i = 0;
while (i < n && someComplexCondition(i)) {
    std::cout << i << " ";
    i++;
}
```

In this case, the loop runs not only based on the value of `i`, but also on the result of a more complex function. This makes `while` loops a good choice when the exit condition depends on multiple variables or non-trivial logic.

**Input Size Consideration**::

- **For Small Inputs**: This is ideal for small inputs where the condition can vary significantly during the iterations.
- **For Large Inputs**: Be cautious with complex conditions when dealing with large inputs, as evaluating the condition on every iteration may add performance overhead.

### 3. Infinite `while` Loops

An infinite `while` loop is a loop that runs indefinitely until an explicit `break` or `return` statement is encountered. This type of loop is typically used in scenarios where the termination condition depends on an external event, such as user input or reaching a specific solution.

```cpp
while (true) {
    // Process some data
    if (exitCondition()) break;
}
```

The loop runs until `exitCondition()` is met, at which point it breaks out of the loop. This structure is useful for algorithms that require indefinite running until a specific event happens.

**Input Size Consideration**:

- **For Small Inputs**: Generally unnecessary for small inputs unless the exit condition is based on dynamic factors.
- **For Large Inputs**: Useful for large inputs when the exact number of iterations is unknown, and the loop depends on a condition that could be influenced by the data itself.

### 4. `do-while` Loop

The `do-while` loop is similar to the `while` loop, but it guarantees that the code block is executed at least once. This is useful when you need to run the loop at least one time regardless of the condition.

```cpp
int i = 0;
do {
    std::cout << i << " ";
    i++;
} while (i < n);
```

In this case, the loop will print `i` at least once, even if `i` starts with a value that makes the condition false. This ensures that the loop runs at least one iteration.

**Input Size Consideration**:

- **For Small Inputs**: Ideal when you need to guarantee that the loop runs at least once, such as with small datasets where the minimum iteration is essential.
- **For Large Inputs**: Suitable for large datasets where the first iteration must occur independently of the condition.

### 5. `while` Loop with Early Exit

The `while` loop can be combined with early exit strategies using `break` or `return` statements to optimize performance, particularly when the loop can terminate before completing all iterations.

```cpp
int i = 0;
while (i < n) {
    if (shouldExitEarly(i)) break;
    std::cout << i << " ";
    i++;
}
```

By including a condition inside the loop that checks for an early exit, you can significantly reduce runtime in cases where processing all elements is unnecessary.

**Input Size Consideration**:

- **For Small Inputs**: It can improve performance when early termination conditions are common or likely.
- **For Large Inputs**: Highly efficient for large datasets, particularly when the early exit condition is met frequently, saving unnecessary iterations.

### 6. Combining `while` with Multiple Conditions

A `while` loop can easily incorporate multiple conditions to create more complex termination criteria. This is particularly useful when multiple variables determine whether the loop should continue.

```cpp
int i = 0;
while (i < n && someOtherCondition()) {
    std::cout << i << " ";
    i++;
}
```

This allows the loop to run based on multiple dynamic conditions, providing more control over the iteration process than a standard `for` loop might offer.

**Input Size Consideration**:

- **For Small Inputs**: A flexible option when the conditions governing the loop may change during execution, even for small datasets.
- **For Large Inputs**: Can be optimized for large datasets by ensuring that the condition checks are efficient and that unnecessary re-evaluations are minimized.

### Optimal `while` Loops for Competitive Programming

Choosing the right type of `while` loop depends on the nature of the input and the complexity of the condition. The following table summarizes the optimal choices for different input sizes:

| Input Size | Best `while` Loop Option                   | Reasoning                                                                                                                  |
| ---------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Small      | Basic `while` Loop                         | Offers straightforward control over iteration with minimal overhead and is easy to implement.                              |
| Small      | `do-while` Loop                            | Ensures at least one execution of the loop, which is crucial for cases where the first iteration is essential.             |
| Large      | `while` with Early Exit                    | Improves performance by terminating the loop early when a specific condition is met, saving unnecessary iterations.        |
| Large      | `while` with Complex Conditions            | Allows dynamic and flexible exit conditions, making it suitable for large datasets with evolving parameters.               |
| Continuous | Infinite `while` Loop with Explicit Breaks | Best for situations where the exact number of iterations is unknown and depends on external factors or dynamic conditions. |
|            |

## Special Loops in C++20 for Competitive Programming

In C++20, several advanced looping techniques have been introduced, each offering unique ways to improve code efficiency and readability. While some of these techniques provide remarkable performance optimizations, not all are well-suited for competitive programming. competitive programmings often involve handling dynamic inputs and generating outputs within strict time limits, so techniques relying heavily on compile-time computation are less practical. This section focuses on the most useful loop structures for competitive programmings, emphasizing runtime efficiency and adaptability to varying input sizes.

### 1. Range-Based Loops with `std::ranges::views`

C++20 introduces `ranges` and `views`, which allow you to create expressive and efficient loops by operating on views of containers without copying data. Views are lazily evaluated, meaning that operations like filtering, transformation, or reversing are applied only when accessed.

**Example**:

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Using views to iterate in reverse
    for (auto elem : vec | std::views::reverse) {
        std::cout << elem << " ";
    }

    return 0;
}
```

**Benefits**:

Efficient and lazy evaluation ensures that operations like reversing or filtering are performed only when needed, rather than precomputing them or creating unnecessary copies of the data. This approach optimizes memory usage and speeds up execution, particularly when working with large datasets.

The syntax is also highly expressive and concise, allowing you to write clear and readable code. This is particularly useful when applying multiple transformations in sequence, as it helps maintain code simplicity while handling complex operations.

**Considerations for competitive programmings**:

Range views are particularly useful when working with large datasets, as they enable efficient processing by avoiding the creation of unnecessary copies and reducing memory overhead. This approach allows for smoother handling of extensive input data, improving overall performance.

Additionally, range views provide clarity and simplicity when dealing with complex operations. They streamline the process of transforming data, making it easier to apply multiple operations in a clean and readable manner, which is especially beneficial in competitive programming scenarios.

### 2. Parallel Loops with `std::for_each` and `std::execution::par`

C++20 enables parallelism in standard algorithms with `std::execution`. Using parallel execution policies, you can distribute loop iterations across multiple threads, which can drastically reduce the execution time for computationally expensive loops. This is especially useful when working with large datasets in competitive programming.

**Example**:

```cpp
#include <execution>
#include <vector>

int main() {
    std::vector<int> vec(1000000, 1);

    std::for_each(std::execution::par, vec.begin(), vec.end(), [](int& elem) {
        elem *= 2;
    });

    return 0;
}
```

**Benefits**:

Parallel loops offer high performance, particularly when dealing with large input sizes that involve intensive computation. By utilizing multiple CPU cores, they significantly reduce execution time and handle heavy workloads more efficiently.

What makes this approach even more practical is that it requires minimal changes to existing code. The parallel execution is enabled simply by adding the execution policy `std::execution::par`, allowing traditional loops to run in parallel without requiring complex modifications.

**Considerations for competitive programmings**:

Parallel loops are highly effective for processing large datasets, making them ideal in competitive programming scenarios where massive inputs need to be handled efficiently. They can dramatically reduce execution time by distributing the workload across multiple threads.

However, they are less suitable for small inputs. In such cases, the overhead associated with managing threads may outweigh the performance gains, leading to slower execution compared to traditional loops.

### 3. `constexpr` Loops

With C++20, `constexpr` has been extended to allow more complex loops and logic at compile time. While this can lead to ultra-efficient code where calculations are precomputed during compilation, this technique has limited utility in competitive programming, where dynamic inputs are a central aspect of the problem. Since competitive programming requires handling varying inputs provided at runtime, `constexpr` loops are generally less useful in this context.

**Example**:

```cpp
#include <array>
#include <iostream>

constexpr std::array<int, 5> generate_squares() {
    std::array<int, 5> arr{};
    for (int i = 0; i < 5; ++i) {
        arr[i] = i * i;
    }
    return arr;
}

int main() {
    constexpr auto arr = generate_squares();
    for (int i : arr) {
        std::cout << i << " ";  // 0 1 4 9 16
    }

    return 0;
}
```

**Benefits**:

Compile-time efficiency allows for faster runtime performance, as all necessary computations are completed during the compilation phase. This eliminates the need for processing during execution, leading to quicker program runs.

This approach is ideal for constant, static data. When all relevant data is known ahead of time, compile-time computation removes the need for runtime processing, providing a significant performance boost by bypassing real-time calculations.

**Considerations for competitive programmings**:

While constexpr loops are not suitable for processing dynamic inputs directly, they can be strategically used to create lookup tables or pre-compute values that are then utilized during runtime calculations. This can be particularly useful in problems involving mathematical sequences, combinatorics, or other scenarios where certain calculations can be predetermined. **However, it's important to balance the use of pre-computed data with memory constraints, as large lookup tables might exceed memory limits in some competitive programming environments**.

### 4. Early Exit Loops

In competitive programming, optimizing loops to exit early when a condition is met can drastically reduce execution time. This approach is especially useful when the solution does not require processing the entire input if an early condition is satisfied.

**Example**:

```cpp
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Early exit if a condition is met
    for (int i = 0; i < vec.size(); ++i) {
        if (vec[i] == 3) break;
        std::cout << vec[i] << " ";
    }

    return 0;
}
```

**Benefits**:

Early exit loops improve efficiency by terminating as soon as a specified condition is met, thus avoiding unnecessary iterations. This approach helps save time, especially when the loop would otherwise continue without contributing to the result.

This technique is particularly useful in search problems. By exiting the loop early when a target value is found, it can significantly enhance performance, reducing the overall execution time.

**Considerations for competitive programmings**:

Early exit loops are highly practical, as they allow a solution to be reached without the need to examine all the data. By cutting down unnecessary iterations, they help reduce execution time, making them particularly useful in scenarios where a result can be determined quickly based on partial input.

### 5. Indexed Loops with Range-Based `for`

While C++ offers powerful range-based `for` loops, there are scenarios where accessing elements by index is essential, especially when the loop logic requires modifying the index or accessing adjacent elements. Range-based `for` loops cannot directly access the index, so indexed loops remain valuable for such cases.

**Example**:

```cpp
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }

    return 0;
}
```

**Benefits**:

Indexed loops offer precise control by providing direct access to elements through their index, giving you full control over how the index changes during iteration. This level of control is crucial when fine-tuning the behavior of the loop.

They are essential when modifying iteration behavior, especially in cases where you need to adjust the index dynamically. This is useful for tasks such as skipping elements or implementing non-linear iteration patterns, allowing for flexible loop management.

**Considerations for competitive programmings**:

Indexed loops are well-suited for dynamic access, offering the flexibility required for more complex iteration logic. This makes them ideal for scenarios where direct control over the loop's behavior is necessary.

However, they are less expressive compared to range-based loops. While they provide detailed control, they tend to be more verbose and less concise than the streamlined syntax offered by range-based alternatives.

### 6. Standard Library Algorithms (`std::for_each`, `std::transform`)

Using standard library algorithms like `std::for_each` and `std::transform` allows for highly optimized iteration and transformation of container elements. These algorithms are highly optimized, making them ideal for competitive programming scenarios where efficiency is crucial.

**Example**:

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::for_each(vec.begin(), vec.end(), [](int& x) { x *= 2; });

    for (const int& x : vec) {
        std::cout << x << " ";
    }

    return 0;
}
```

**Benefits**:

Standard library algorithms are highly optimized for performance, often surpassing the efficiency of manually written loops. Their internal optimizations make them a powerful tool for handling operations in a time-efficient manner.

Additionally, these functions are concise and clear, providing a clean and expressive syntax to apply operations on containers. This simplicity enhances code readability while maintaining high performance, making them ideal for competitive programming.

**Considerations for competitive programmings**:

Standard library algorithms are great for transformation tasks, allowing you to apply operations on container elements with minimal code. They maximize efficiency while keeping the implementation simple and concise, making them particularly effective for handling transformations in competitive programming scenarios.

### Summary Table of Useful Loop Techniques for competitive programmings

| Technique                                 | Best Use Case                            | Efficiency Considerations                                                          |
| ----------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------- |
| `std::ranges::views`                      | Transforming or filtering large datasets | Lazily evaluated operations reduce memory overhead and improve runtime efficiency. |
| Parallel Loops with `std::execution::par` | Large computational tasks                | Parallelism significantly improves performance for large, independent tasks.       |
| Early Exit Loops                          | Search or conditional exit problems      | Avoids unnecessary iterations, improving efficiency in scenarios with early exits. |
| Indexed Loops                             | Precise control over iteration           | Offers flexibility and control for complex iteration logic or index manipulation.  |
| Standard Library Algorithms               | Applying transformations or actions      | Well-optimized algorithms that simplify code and improve performance.              |

### Techniques Not Recommended for competitive programmings

| Technique         | Reasoning                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `constexpr` Loops | Compile-time only, cannot handle dynamic input, thus impractical for runtime competitive programming problems. |

## Dynamic Programming

Dynamic Programming is a different way of thinking when it comes to solving problems. Programming itself is already a different way of thinking, so, to be honest, I can say that Dynamic Programming is a different way within a different way of thinking. And, if you haven't noticed yet, there is a concept of recursion trying to emerge in this definition.

The general idea is that you, dear reader, should be able to break a large and difficult problem into small and easy pieces. This involves storing and reusing information within the algorithm as needed.

It is very likely that you, kind reader, have been introduced to Dynamic Programming techniques while studying algorithms without realizing it. So, it is also very likely that you will encounter, in this text, algorithms you have seen before without knowing they were Dynamic Programming.

My intention is to break down the Dynamic Programming process into clear steps, focusing on the solution algorithm, so that you can understand and implement these steps on your own whenever you face a problem in technical interviews, production environments, or programming competitive programmings. Without any hesitation, I will try to present performance tips and tricks in C++. However, this should not be considered a limitation; we will prioritize understanding the algorithms before diving into the code, and you will be able to implement the code in your preferred programming language.

I will be using functions for all the algorithms I study primarily because it will make it easier to measure and compare the execution time of each one, even though I am aware of the additional computational overhead associated with function calls. After studying the problems in C++ and identifying the solution with the lowest complexity, eventually, we will also explore the best solution in C. Additionally, whenever possible, we will examine the most popular solution for the problem in question that I can find online.

## There was a hint of recursion sneaking in

Some say that Dynamic Programming is a technique to make recursive code more efficient. If we look at Dynamic Programming, we will see an optimization technique that is based on recursion but adds storage of intermediate results to avoid redundant calculations. _Memoization and tabulation are the two most common Dynamic Programming techniques_, each with its own approach to storing and reusing the results of subproblems:

- **Memoization (Top-Down)**: _This technique is recursive in nature_. It involves storing the results of expensive function calls and returning the cached result when the same inputs occur again. This approach can be seen as an optimization of the top-down recursive process.
- **Tabulation (Bottom-Up**): _Tabulation takes an iterative approach, solving smaller subproblems first and storing their solutions in a table (often an array or matrix)_. It then uses these stored values to calculate the solutions to larger subproblems, gradually building up to the final solution. The iterative nature of tabulation typically involves using loops to fill the table in a systematic manner.

At this point, we can take a look at two examples using Python as pseudocode, since many of my students feel comfortable with Python. First, an example of Dynamic Programming with memoization.

### Example 2: Memoization

Let's revisit the Fibonacci sequence problem: finding the nth number in the Fibonacci sequence. This time, we'll use Dynamic Programming with memoization. Flowchart 2 illustrates this approach.

![]({{ site.baseurl }}/assets/images/recursive-memo.jpg)
_Flowchart 2 - Recursive Fibonacci nth algorithm with memoization_{: class="legend"}

From Flowchart 2, we can derive the following Python code:

```python
# dictionary creation
memo = {} #just for style
# function
def fibonacci_memo(n, memo):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

_Code Fragment 2 - Memoization Function for the nth Term_{: class="legend"}

The `fibonacci_memo()` function is then defined to calculate the nth Fibonacci number using the stored values in a dictionary. Let's analyze the `fibonacci_memo()` code in detail.

#### Function Definition and Initialization

The `fibonacci_memo()` function begins by:

```python
memo = {} #just for style
def fibonacci_memo(n, memo):
```

In this code fragment, there is a dictionary named memo declared as `memo = {}`. It will be used to _store the results of previous Fibonacci calculations, preventing redundant calculations and improving efficiency_ (this is memoization!), which brings us to the consideration of the base case within recursion.

#### Base Case

```python
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
```

The line `if n in memo` checks if the value of $𝑛$ has already been calculated and stored in the `memo` dictionary. If so, it returns the stored value, avoiding recalculation. On the other hand, `if n <= 1` handles the base cases of the Fibonacci sequence:

- `fibonacci(0) = 0`
- `fibonacci(1) = 1`

#### Recursive Step and Memoization

```python
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

The expression `fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)` initiates recursive calls to determine the $(n-1)$th and $(n-2)$th Fibonacci numbers. This is the fundamental recursive relationship in the Fibonacci Sequence: each number is the sum of the two preceding ones.

The `memo` dictionary is the key to memoization. Before making the recursive calls, the function checks if the results for $n-1$ and $n-2$ are already stored in `memo`. If so, those stored values are used directly, avoiding redundant calculations. If not, the recursive calls are made, and the results are stored in `memo` for future reference.

The calculated result (`fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)`) is assigned to `memo[n]`, storing it for potential reuse later.

Finally, return `memo[n]` returns the calculated (and now memoized) value for the nth Fibonacci number.

From the perspective of Dynamic Programming, the `fibonacci_memo` function employs a divide-and-conquer strategy, breaking down the calculation of the nth Fibonacci number into smaller subproblems (calculating the ($n-1$)th and ($n-2$)th numbers). It leverages a dictionary, memo, to store and retrieve the results of these subproblems. This approach eliminates redundant computations, enhancing efficiency, and before calculating the Fibonacci value for a given $n$, the function checks if the result is already stored in the `memo` dictionary. If it is, it reuses that result, saving computation time. Finally, the function ensures that each subproblem is solved only once, resulting in more efficiency compared to the simple recursive approach.

The last statement of the previous paragraph requires reflection. I am considering performance in this statement only in terms of computation time. Performance can also be considered in relation to memory usage, energy consumption, and any other factor that is interesting or important for a given problem. Keep this in mind whenever I state that performance has improved in this text.

Performance can be evaluated through complexity analysis. When analyzing the complexity of an algorithm, we often refer to its time complexity and space complexity. _Time complexity refers to the amount of time an algorithm takes to run as a function of the size of its input. Space complexity refers to the amount of memory an algorithm uses as a function of the size of its input_. Both are crucial aspects of performance.

For example, the naive recursive Fibonacci algorithm has a time complexity of $O(2^n)$ because it makes an exponential number of calls. With memoization, the time complexity is reduced to $O(n)$ since each Fibonacci number up to $n$ is computed only once. The space complexity also becomes $O(n)$ due to the storage of computed values in the `memo` dictionary.

Now, you might wonder: How many recursive calls does a memoized Fibonacci function actually make?

To figure this out, let's see how memoization changes the usual recursion tree:

- **Base Case**: If the Fibonacci number for $n$ is already stored in our memoization cache, or if $n$ is $0$ or $1$, the function returns directly without any further calls.
- **Memoization Check**: If $n$ isn't in the cache, the function makes two recursive calls: `fibonacci_memo(n-1, memo)` and `fibonacci_memo(n-2, memo)`.
- **The Memoization Effect**: The very first time we call `fibonacci_memo(n, memo)` with a new value of $n$, it will keep making recursive calls until it hits the base cases. Crucially, each Fibonacci number, once computed, is stored in the cache, it gets stored in the cache. Subsequent calls with the same value of n retrieve the stored result, circumventing further recursive calls.

#### Calculating the Number of Recursive Calls

To understand the efficiency of our memoized Fibonacci function, we need to calculate the number of recursive calls made during its execution. Memoization significantly reduces the number of redundant calls, resulting in a more efficient algorithm. Below, we break down the process:

- **Initial Call**: We start the whole process with a single call to `fibonacci_memo(n, memo)`.
- **Recursive Expansion**: For every new $n$ value we encounter, the function branches out into calls for `fibonacci_memo(n-1, memo)` and `fibonacci_memo(n-2, memo)`.
- **Memoization Storage**: Each calculated value is stored, hence any future calls with the same $n$ don't create new branches.
- **Counting Unique Calls**: Because of memoization, we only need to calculate each Fibonacci number once. This means the total number of recursive calls is roughly equal to the number of unique Fibonacci numbers up to $n$.

In conclusion, while a naive Fibonacci implementation would have an exponential number of calls, memoization brings this down significantly. We have approximately $n+1$ calls to calculate Fibonacci numbers from $0$ to $n$, resulting in linear growth. This is a significant improvement over the exponential growth (roughly $2^n$ calls) in the naive recursive approach. The key to this efficiency is that each unique Fibonacci number is calculated only once and then stored for future use, making the process much more efficient.

To calculate the total number of recursive calls made when computing the nth Fibonacci number using our memoized function, we can use the following recurrence relation:

$$
T(n) = \begin{cases}
1 & \text{if } n \text{ is not in memo} \
0 & \text{if } n \text{ is in memo}
\end{cases}
$$

Where $T(n)$ represents the number of new recursive calls needed to compute the nth Fibonacci number using our memoized function.

This formula accounts for: a single new call if the value isn't memoized yet, $(1)$, and, no new calls if the value is already in the memo, $(0)$.

It's important to note that this formula gives us the number of new calls for each Fibonacci number, not the cumulative number of calls. To illustrate how this works for $n = 10$, let's calculate the number of recursive calls $T(10)$. We'll start with the base values and then calculate the subsequent values up to $T(10)$.

Assuming that no values are initially memoized:

$$
\begin{aligned}
T(0) &= 1 \text{ (not in memo)} \
T(1) &= 1 \text{ (not in memo)} \
T(2) &= 1 \text{ (not in memo)} \
T(3) &= 1 \text{ (not in memo)} \
T(4) &= 1 \text{ (not in memo)} \
T(5) &= 1 \text{ (not in memo)} \
T(6) &= 1 \text{ (not in memo)} \
T(7) &= 1 \text{ (not in memo)} \
T(8) &= 1 \text{ (not in memo)} \
T(9) &= 1 \text{ (not in memo)} \
T(10) &= 1 \text{ (not in memo)} \
\end{aligned}
$$

Therefore, the total number of new recursive calls to compute fibonacci_memo(10) is:

$$\sum_{i=0}^{10} T(i) = 11$$

Each value of $T(n)$ represents a new recursive call to compute `fibonacci_memo(n)`. And we have only $11$ new calls in total, demonstrating the efficiency of memoization compared to the non-memoized version which would have $177$ calls.

#### Time and Space Complexity

We must proceed to the complexity analysis, focusing on the Big O notation, of the fibonacci_memo function, which uses memoization to calculate Fibonacci numbers. Let's analyze its time and space complexity. The key to understanding the time complexity is that each unique value of $n$ is calculated only once and then stored in `memo`.

So, as there are $n$ unique values (from $0$ to $n$), for each value of $n$, the function executes a fixed amount of operations (checking, adding, and retrieving values from `memo`). Therefore, the total time complexity of the function is $O(n)$, since each Fibonacci number up to $n$ is computed and stored once, and only once.

The space complexity is determined by the additional storage used by the memoization dictionary (`memo`). The dictionary `memo` can contain up to $n$ entries (one for each Fibonacci number up to $n$). Consequently, the space complexity is also $O(n)$ due to the storage needs of the `memo` dictionary.

We are now ready to study Dynamic Programming with Tabulation.

### Example 3: Fibonacci with Tabulation

Now, let's explore an example of Dynamic Programming using the tabulation technique:

![]({{ site.baseurl }}/assets/images/interactive-fibbo.jpg)
_Flowchart 3 - Interactive Fibonacci nth algorithm with tabulation_{: class="legend"}

Here is the function `fibonacci_tabulation` defined to calculate the nth Fibonacci number using tabulation, utilizing Python in a manner similar to pseudocode:

```python
def fibonacci_tabulation(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

_Code Fragment 3 - Tabulation Function for the nth Term_{: class="legend"}

Unlike the previous recursive function, this function uses an iterative approach known as tabulation, a bottom-up Dynamic Programming technique. In this implementation, the `fibonacci_tabulation()` function uses a list, `dp`, to store the results of all subproblems, building the solution from the bottom up. It is important to note that data is being stored in this tabulation approach.

Indeed! But look closely. The `fibonacci_tabulation()` function is an example of tabulation, not memoization, due to the distinct manner in which subproblems are solved and their solutions stored.

_Tabulation is a bottom-up approach to Dynamic Programming where you solve all subproblems first and store their solutions in a data structure, usually a table, array, list, or tree_. The solution to the larger problem is then built from these smaller solutions by traversing the data structure from the bottom up. _This implies an iterative resolution process_. The subproblems are solved iteratively, starting from the smallest until the larger problem is reached. In this case, recursion is irrelevant.

#### Function Definition and Initialization

```python
def fibonacci_tabulation(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
```

_Code Fragment 3A - Tabulation Function Initialization_{: class="legend"}

- `if n <= 1: return n`: This handles the base cases of the Fibonacci Sequence. If $n$ is $0$ or $1$, it directly returns n because:

  - `fibonacci_tabulation(0)` $= 0$
  - `fibonacci_tabulation(1)` $= 1$

- `dp = [0] * (n + 1)`: This initializes a list dp with `n+1` zeros. This list will store the Fibonacci numbers up to $n$.

- `dp[1] = 1`: This declaration sets the second element of `dp` to $1$, since `fibonacci_tabulation(1)` $= 1$.

#### Iteration and Calculation

```python
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

_Code Fragment 3B - Tabulation Function Iteration_{: class="legend"}

-`for i in range(2, n + 1)`: This loop starts from $2$ and iterates up to $n$. -`dp[i] = dp[i-1] + dp[i-2]`: This calculates the ith Fibonacci number by summing the previous two Fibonacci numbers (i.e., `fibonacci_tabulation(i-1)` and `fibonacci_tabulation(i-2)`) and stores it in the `dp` list at index $i$. -`return dp[n]`: After the loop completes, the function returns the nth Fibonacci number stored in `dp[n]`.

#### Flow Explanation

Let's try with the tenth Fibonacci number. When `fibonacci_tabulation(10)` is called, it checks if $10 <= 1$. It is not, so it proceeds.

Initializes the `dp` list with zeros: `dp = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.
Iterates from $2$ to $10$:

- `For i = 2: dp[2] = dp[1] + dp[0]` $= 1 + 0 = 1$
- `For i = 3: dp[3] = dp[2] + dp[1]` $= 1 + 1 = 2$
- `For i = 4: dp[4] = dp[3] + dp[2]` $= 2 + 1 = 3$
- This continues until $i = 10$.

After the loop, `dp` is `[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]`.
Returns `dp[10]`, which is $55$. And this is the moment when we stop to celebrate.

#### Time and Space Complexity

The time complexity analysis of the function `fibonacci_tabulation()` begins with the initial check to see if `n` is less than or equal to $1$, which is a constant time operation, $O(1)$. If `n` is greater than $1$, the function initializes a list `dp` with `n + 1` elements, which takes $O(n)$ time. After this, the function sets `dp[1]` to $1$, another constant time operation, $O(1)$.

Assuming a complete implementation that iterates from $2$ to `n` to fill in the `dp` array, the total time complexity would be $O(n)$, as each Fibonacci number up to `n` is computed and stored exactly once.

The space complexity is dominated by the list `dp` with `n + 1` elements, which requires $O(n)$ space. Therefore, with the complete implementation in mind, the function achieves an efficient computation with both time and space complexities of $O(n)$.

The time complexity of the `fibonacci_tabulation()` function is $O(n)$, as each Fibonacci number up to $n$ is computed and stored exactly once. The space complexity is also $O(n)$ due to the `dp` list with n + 1 elements. While this complexity analysis covers the basic aspects of the function's efficiency, there are additional considerations and potential optimizations that could further enhance its performance:

- **Space Optimization**: The current implementation uses $O(n)$ space, but we only need the last two Fibonacci numbers to calculate the next one. We could optimize the space usage to $O(1)$ by using only two variables instead of an array.
- **Modular Arithmetic**: For very large $n$, the Fibonacci numbers can grow extremely large. In some applications, we might only need the result modulo some number (e.g., 10^9 + 7). Implementing modular arithmetic can prevent integer overflow and improve performance.
- **Matrix Exponentiation**: For extremely large $n$, we can use matrix exponentiation to calculate Fibonacci numbers in $O(log n)$ time, though this involves a more complex implementation.
- **Parallelization**: For certain ranges of Fibonacci numbers, we could potentially use parallel processing to speed up computations, especially on multi-core systems.
- **Memoization trade-offs**: While our tabulation approach is bottom-up, in some scenarios, a top-down memoized approach might be more suitable, especially if we don't always need all Fibonacci numbers up to $n$.

These considerations invite deeper exploration into algorithm design, mathematical properties of Fibonacci numbers, and advanced programming techniques. Each optimization presents its own trade-offs in terms of implementation complexity, memory usage, and computational efficiency.

### There is more between heaven and earth, Mr. Shakespeare

Memoization and tabulation are the most common techniques in Dynamic Programming; however, they are not the only techniques.

- **Dynamic Programming with State Compression**: The goal is to reduce the space needed to store the results of the subproblems by keeping only the states relevant to calculating the final solution.
- **Dynamic Programming with Sliding Window**: Maintains only the results of the most recent subproblems in a fixed-size window, useful when the solution depends only on a limited number of previous subproblems.
- **Dynamic Programming with Decision Tree**: Represents the subproblems and their relationships in a decision tree, allowing a clear visualization of the problem structure and the decisions to be made.

Let's see how far we get in this text. As I write this, I still have no idea.

## Now I realize: C++, where is C++?

Throughout our exploration of Dynamic Programming concepts, we've been using Python as a form of pseudocode. Its versatility and simplicity have served us well, especially considering that many of my students are already familiar with it. Python's readability has made it an excellent choice for introducing and illustrating algorithmic concepts. However, as we progress into more advanced territory, it's time to acknowledge that Python, despite its strengths, isn't the most suitable language for high-performance applications or programming competitive programmings.

With this in mind, we're going to transition to using **C++ 20** as our primary language moving forward. C++ offers superior performance, which is crucial when dealing with computationally intensive tasks often found in competitive programming scenarios. It also provides more direct control over memory management, a feature that can be essential when optimizing algorithms for speed and efficiency. Additionally, we'll occasionally use data structures compatible with **C 17** within our **C++ 20** environment, ensuring a balance between modern features and broader compatibility.

For our development environment, we'll be using Visual Studio Community Edition. This robust IDE will allow us to write, compile, and evaluate our C++ code effectively. It offers powerful debugging tools and performance profiling features, which will become increasingly valuable as we delve into optimizing our algorithms.

Despite this shift, we won't be discarding the work we've done so far. To maintain consistency and provide a bridge between our previous discussions and this new approach, I'll be converting the functions we originally wrote in Python to C++.

As we make this transition, we'll gradually introduce C++ specific optimizations and techniques, broadening your understanding of Dynamic Programming implementation across different language paradigms. I hope this approach will equip you with both a solid conceptual foundation and the practical skills needed for high-performance coding.

### Example 4: Fibonacci in C++ using `std::vectors`

Let's begin with a straightforward, naive implementation in **C++20**, following the flow and data structures of the Python functions provided earlier.

```Cpp
#include <iostream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <functional>

// Recursive function to calculate Fibonacci
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

// Recursive function with memoization to calculate Fibonacci
int fibonacci_memo(int n, std::unordered_map<int, int>& memo) {
    if (memo.find(n) != memo.end()) {
        return memo[n];
    }
    if (n <= 1) {
        return n;
    }
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo);
    return memo[n];
}

// Iterative function with tabulation to calculate Fibonacci
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    std::vector<int> dp(n + 1, 0);
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

// Function to measure execution time
template <typename Func, typename... Args>
long long measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return duration.count();
}

// Function to calculate average execution time
template <typename Func, typename... Args>
long long average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    for (int i = 0; i < iterations; ++i) {
        total_time += measure_time(func, std::forward<Args>(args)...);
    }
    return total_time / iterations;
}

int main() {

    const int iterations = 1000;
    std::vector<int> test_cases = { 10, 20, 30 };

    for (int n : test_cases) {
        std::cout << "Calculating Fibonacci(" << n << ")\n";

        // Calculation and average time using the simple recursive function
        long long avg_time_recursive = average_time(fibonacci, iterations, n);
        std::cout << "Average time for recursive Fibonacci: " << avg_time_recursive << " ns\n";

        // Calculation and average time using the memoization function
        std::unordered_map<int, int> memo;
        auto fibonacci_memo_wrapper = [&memo](int n) { return fibonacci_memo(n, memo); };
        long long avg_time_memo = average_time(fibonacci_memo_wrapper, iterations, n);
        std::cout << "Average time for memoized Fibonacci: " << avg_time_memo << " ns\n";

        // Calculation and average time using the tabulation function
        long long avg_time_tabulation = average_time(fibonacci_tabulation, iterations, n);
        std::cout << "Average time for tabulated Fibonacci: " << avg_time_tabulation << " ns\n";

        std::cout << "-----------------------------------\n";
    }

    return 0;
}
```

_Code 1 - Running `std::vector` and tail recursion_{: class="legend"}

The Code 1 demonstrates not only our Fibonacci functions but also two functions for calculating execution time (`long long measure_time(Func func, Args&&... args)` and `long long measure_time(Func func, Args&&... args)`). From this point forward, I will be using this code model to maintain consistent computational cost when calculating the average execution time of the functions we create. This approach will ensure that our performance measurements are standardized across different implementations, allowing for more accurate comparisons as we explore various Dynamic Programming techniques.

Now, the attentive reader will agree with me: we must to break this code down.

#### The Recursive Function

Let's start with `fibonacci(int n)`, the simple and pure tail recursive function.

```Cpp
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}
```

_Code Fragment 1A - C++ Tail Recursion Function_{: class="legend"}

This is a similar C++ recursive function to the one we used to explain recursion in Python. Perhaps the most relevant aspect of `fibonacci(int n)` is its argument: `int n`. Using the `int` type limits our Fibonacci number to $46$. Especially because the `int` type on my system, a 64-bit computer running Windows 11, is limited, by default, to storing a maximum value of $2^31 - 1 = 2,147,483,647$, and the $46$th Fibonacci number is $1,836,311,903$. The 47th Fibonacci number will be bigger will be bigger than `int` capacity.

The next function is the C++ memoization version:

#### The Dynamic Programming Function Using Memoization

```Cpp
// Recursive function with memoization to calculate Fibonacci
int fibonacci_memo(int n, std::unordered_map<int, int>& memo) {
    if (memo.find(n) != memo.end()) {
        return memo[n];
    }
    if (n <= 1) {
        return n;
    }
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo);
    return memo[n];
}
```

_Code Fragment 2A - C++ Memoization Function_{: class="legend"}

Let's highlight the `std::unordered_map<int, int>& memo` in function arguments. The argument `std::unordered_map<int, int>& memo` in C++ is used to pass a reference to an unordered map (hash table) that maps integers to integers. Breaking it down we will have:

> `std::unordered_map` is a template class provided by the C++ Standard Library (STL) that implements a hash table. The template parameters `<int, int>` specify that the keys and values stored in the unordered map are both integers.

The ampersand (`&`) indicates that the argument is a reference. This means that the function will receive a reference to the original unordered map, rather than a copy of it. _Passing by reference is efficient because it avoids copying the entire map, which could be expensive in terms of time and memory, especially for large maps_. Pay attention: Thanks to the use of `&`, all changes made to the map inside the function will affect the original map outside the function. Finally `memo` is the identifier of the parameter which type is `std::unordered_map<int, int>`.In the context of memoization (hence the name `memo` we used earlier), this unordered map is used to store the results of previously computed values to avoid redundant calculations.

> One `unordered_map` in C++ is quite similar to Python's dict in terms of functionality. Both provide an associative container that allows for efficient key-value pair storage and lookup. The `std::unordered_map` is a template class and a C++ only construct implemented as a hash table. _Unordered maps store key-value pairs and provide average constant-time complexity for insertion, deletion, and lookup operations, thanks to their underlying hash table structure_. They grow dynamically as needed, managing their own memory, which is freed upon destruction. Unordered maps can be passed to or returned from functions by value and can be copied or assigned, performing a deep copy of all stored elements.

> Unlike arrays, unordered maps do not decay to pointers, and you cannot get a pointer to their internal data. Instead, unordered maps maintain an internal hash table, which is allocated dynamically by the allocator specified in the template parameter, usually obtaining memory from the freestore (heap) independently of the object's actual allocation. This makes unordered maps efficient for fast access and manipulation of key-value pairs, though they do not maintain any particular order of the elements.

> Unordered maps do not require a default constructor for stored objects and are well integrated with the rest of the STL, providing `begin()`/`end()` methods and the usual STL typedefs. When reallocating, unordered maps rehash their elements, which involves reassigning the elements to new buckets based on their hash values. This rehashing process can involve copying or moving (in C++11 and later) the elements to new locations in memory.

> Rehashing, is the process used in `std::unordered_map` to maintain efficient performance by redistributing elements across a larger array when the load factor (the number of elements divided by the number of buckets) becomes too high. The rehashing process involves determining the new size, allocating a new array of buckets to hold the redistributed elements, rehashing elements by applying a hash function to each key to compute a new bucket index and inserting the elements into this new index, and finally, updating the internal state by updating internal pointers, references, and variables, and deallocating the old bucket array. Rehashing in `std::unordered_map` is crucial for maintaining efficient performance by managing the load factor and ensuring that hash collisions remain manageable.

Overall, `std::unordered_map` is a versatile and efficient container for associative data storage, offering quick access and modification capabilities while seamlessly integrating with the C++ Standard Library and, for our purposes, very similar to Python's dictionary

The `fibonacci_memo(int n, std::unordered_map<int, int>& memo)` function works just like the Python function we explained before with the same complexity, $O(n)$, for space and time. That said we can continue to `fibonacci_tabulation(int n)`.

Memoization offers significant advantages over the simple recursive approach when implementing the Fibonacci sequence. The primary benefit is improved time complexity, reducing it from exponential $O(2^n)$ to linear $O(n)$. This optimization is achieved by storing previously computed results in a hash table (memo), eliminating redundant calculations that plague the naive recursive method. This efficiency becomes increasingly apparent for larger $n$ values, where the simple recursive method's performance degrades exponentially, while the memoized version maintains linear time growth. Consequently, memoization allows for the computation of much larger Fibonacci numbers in practical time frames.

#### The Dynamic Programming Function Using Tabulation

The `fibonacci_tabulation(int n)`, which uses a `std::vector`, was designed to be as similar as possible to the tabulation function we studied in Python.

```CPP
// Iterative function with tabulation to calculate Fibonacci
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    std::vector<int> dp(n + 1, 0);
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

_Code 1C - C++ Tabulation Function_{: class="legend"}

> The `std::vector` is a template class and a C++ only construct implemented as a dynamic array. _Vectors grow and shrink dynamically, automatically managing their memory, which is freed upon destruction. They can be passed to or returned from functions by value and can be copied or assigned, performing a deep copy of all stored elements_.
>
> Unlike arrays, vectors do not decay to pointers, but you can explicitly get a pointer to their data using `&vec[0]`. Vectors maintain their size (number of elements currently stored) and capacity (number of elements that can be stored in the currently allocated block) along with the internal dynamic array. This internal array is allocated dynamically by the allocator specified in the template parameter, _usually obtaining memory from the freestore (heap) independently of the object's actual allocation_. Although this can make vectors less efficient than regular arrays for small, short-lived, local arrays, vectors do not require a default constructor for stored objects and are better integrated with the rest of the STL, providing `begin()`/`end()` methods and the usual STL typedefs. When reallocating, vectors copy (or move, in C++11) their objects.

Besides the `std::vector` template type, the time and space complexity are the same, $O(n)$, we found in Python version. What left us with the generic part of Code 1: Evaluation.

#### Performance Evaluation and Support Functions

All the effort we have made so far will be useless if we are not able to measure the execution times of these functions. In addition to complexity, we need to observe the execution time. This time will depend on the computational cost of the structures used, the efficiency of the compiler, and the machine on which the code will be executed. I chose to find the average execution time for calculating the tenth, twentieth, and thirtieth Fibonacci numbers. To find the average, we will calculate each of them 1000 times. For that, I created two support functions:

```Cpp
// Function 1: to measure execution time
template <typename Func, typename... Args>
long long measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return duration.count();
}

// Function 2: to calculate average execution time
template <typename Func, typename... Args>
long long average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    for (int i = 0; i < iterations; ++i) {
        total_time += measure_time(func, std::forward<Args>(args)...);
    }
    return total_time / iterations;
}
```

_Code Fragment 7 - Support Functions for Time Execution Measurement_{: class="legend"}

Let's initiate with function `long long average_time(Func func, int iterations, Args&&... args)`. This function is a template function designed to measure the execution time of a given function `func` with arbitrary arguments `Args&&... args`. It returns the time taken to execute the function in nanoseconds. Let's break down each part of this function to understand how it works in detail.

```cpp
template <typename Func, typename... Args>
```

The keyword `template` in `measure_time` declaration indicates that `measure_time` is a template function, which means it can operate with generic types.

> A template is a C++ language feature that allows functions and classes to operate with generic types, enabling code reuse and type safety, allowing the creation of functions and classes that can work with any data type without being rewritten for each specific type. This is achieved by defining a blueprint that specifies how the function or class should operate with type parameters that are provided when the template is instantiated. The advantage of templates is their ability to provide high levels of abstraction while maintaining performance, as template code is resolved at compile time, resulting in optimized and type-safe code. This leads to more flexible and reusable code structures, reducing redundancy and the potential for errors, and allowing developers to write more generic and maintainable code.

The first argument, `typename Func` specifies that the first template parameter, `Func`, can be any callable type, such as functions, function pointers, lambdas, or functors. When `typename Func` is specified in a template definition, it indicates that the template will accept a callable entity as a parameter. The use of `typename` in this context ensures that the template parameter is interpreted as a type, enabling the compiler to correctly process the callable type during instantiation. I am using `Func` to call the function whose execution time will be measured.

The last argument, `typename... Args`: This is a variadic template parameter, allowing the function to accept any number of additional arguments of any types.

> The `typename... Args` declaration is used in C++ templates to define a variadic template parameter, which allows a template to accept an arbitrary number of arguments of any types. When `typename... Args` is specified, it indicates that the template can handle a variable number of parameters, making it highly flexible and adaptable. This is particularly useful for functions and classes that need to operate on a diverse set of inputs without knowing their types or number in advance.

The line `auto start = std::chrono::high_resolution_clock::now();` is a crucial component in precise time measurement in C++. It utilizes the C++ Standard Library's `<chrono>` library, which provides a set of time utilities.

> The `std::chrono::high_resolution_clock` is a clock type that represents the clock with the smallest tick period available on the system. The `now()` function is a static member function of this clock class that returns the current time point. By calling `now()`, we capture the exact moment in time when this line is executed.
>
> The auto keyword is used here for type inference, allowing the compiler to automatically deduce the type of the start variable. In this case, start will be of type `std::chrono::time_point<std::chrono::high_resolution_clock>, which represents a point in time as measured by the high-resolution clock. This time point can later be compared with another time point (typically captured after the execution of the code we want to measure) to calculate the duration of the executed code.

In our case, we will compare the `start` time with the `end` time acquired by `auto end = std::chrono::high_resolution_clock::now();`. Between this two lines is `func(std::forward<Args>(args)...);`.

The line `func(std::forward<Args>(args)...);` is a key component of the `measure_time` function. In this context, it serves to execute the function `func` that we aim to measure, while passing along all arguments provided to `measure_time`. This line appears between the two time measurements (`start` and `end`), allowing us to capture the execution time of func with its given arguments. The use of `std::forward` and parameter pack expansion allows `measure_time` to work with functions of any signature, making it a flexible timing utility.

In the context of template functions like `measure_time`, `func` typically represents a callable object. This can be a regular function, a function pointer, a lambda expression, or a function object (functor). The exact type of `func` is deduced by the compiler based on the argument passed to `measure_time`.

> `std::forward` is a utility function template defined in the `<utility>` header of the C++20 Standard Library. Its primary use is in implementing perfect forwarding. `std::forward` preserves the value category (`lvalue` or `rvalue`) of a template function argument when passing it to another function. This allows the called function to receive the argument with the same value category as it was originally passed.
>
> Perfect forwarding allows a function template to pass its arguments to another function while retaining the `lvalue`/`rvalue` nature of the arguments. This is achieved by declaring function parameters as forwarding references (`T&&`) and using `std::forward` when passing these parameters to other functions.

For example:

```cpp
template<class T>
void wrapper(T&& arg) {
    foo(std::forward<T>(arg));
}
```

In this example, wrapper can accept both `lvalues` and `rvalues`, and will pass them to foo preserving their original value category. The combination of forwarding references and `std::forward` enables the creation of highly generic code that can work efficiently with a wide variety of argument types and value categories. This is particularly useful in library design and when creating wrapper functions or function templates that need to preserve the exact characteristics of their arguments when forwarding them to other functions.

However, all this code will not work if we don't take the necessary precautions in the function signature.

```Cpp
long long measure_time(Func func, Args&&... args) {
```

In the context of a template function, `Args&&... args` is often used to perfectly forward these arguments to another function, preserving their value categories (`lvalues` or `rvalues`). The use of `typename...` ensures that each parameter in the pack is treated as a type, enabling the compiler to correctly process each argument during template instantiation.

> An `lvalue` (locator value) represents an object that occupies a specific location in memory (i.e., has an identifiable address). `lvalues` are typically variables or dereferenced pointers. They can appear on the left-hand side of an assignment expression, hence the name `lvalue`.
>
> An `rvalue` (read value) represents a temporary value or an object that does not have a persistent memory location. rvalues are typically literals, temporary objects, or the result of expressions that do not persist. They can appear on the right-hand side of an assignment expression.
>
> C++11 introduced `rvalue` references to enhance performance by enabling move semantics. An rvalue reference is declared using `&&`, allowing functions to distinguish between copying and moving resources. This is particularly useful for optimizing the performance of classes that manage resources such as dynamic memory or file handles.

The return type of the function, `long long`, represents the duration of the function execution in nanoseconds. I choose a `long long` integer because I have no idea how long our Dynamic Programming functions will take to compute, and I wanted to ensure a default function that can be used for all problems we will work on. The maximum value that can be stored in a `long long` type in C++ is defined by the limits of the type, which are specified in the `<climits>` header. For a signed `long long` type, the maximum value is $2^{63} - 1 = 9,223,372,036,854,775,807$.

The function `measure_time` arguments are:

- `Func func`: The callable entity whose execution time we want to measure.
- `Args&&... args: A parameter pack representing the arguments to be forwarded to the callable entity. The use of && indicates that these arguments are perfect forwarded, preserving their value category (`lvalue`or`rvalue`).

As we saw before, the the body of function `measure_time` starts with:

```Cpp
auto start = std::chrono::high_resolution_clock::now();
```

Where `auto start` declares a variable `start` to store the starting time point and
`std::chrono::high_resolution_clock::now()` retrieves the current time using a high-resolution clock, which provides the most accurate and precise measurement of time available on the system. The `std::chrono::high_resolution_clock::now()` returns a `time_point` object representing the current point in time.

> In C++, a `time_point` object is a part of the `<chrono>` library, which provides facilities for measuring and representing time. A `time_point` object represents a specific point in time relative to a clock. It is templated on a clock type and a duration, allowing for precise and high-resolution time measurements. The Clock is represented by a `clock type`, and can be system_clock, steady_clock, or high_resolution_clock. The clock type determines the epoch (the starting point for time measurement) and the tick period (the duration between ticks).

Following we have the function call:

```Cpp
func(std::forward<Args>(args)...);
```

`func`: Calls the function or callable entity passed as the `func` parameter while `std::forward<Args>(args)...` forwards the arguments to the function call. This ensures that the arguments are passed to the called function, `func` with the same value category (`lvalue` or `rvalue`) that they were passed to `measure_time`.

We measure the time and store it in `start`, then we call the function. Now we need to measure the time again.

```Cpp
auto end = std::chrono::high_resolution_clock::now();
```

In this linha `auto end` declares a variable `end` to store the ending time point while `std::chrono::high_resolution_clock::now()` retrieves the current time again after the function `func` has completed execution. Finally we can calculate the time spent to call the function `func`.

```Cpp
std::chrono::duration<long long, std::nano> duration = end - start;
```

Both the `start` and `end` variables store a `time_point` object. `std::chrono::duration<long long, std::nano>` represents a duration in nanoseconds. `end - start` calculates the difference between the ending and starting time points, which gives the duration of the function execution.

> In C++, the `<chrono>` library provides a set of types and functions for dealing with time and durations in a precise and efficient manner. One of the key components of this library is the std::chrono::duration class template, which represents a time duration with a specific period.

The `std::chrono::duration<long long, std::nano>` declaration can be break down as:

- `std::chrono`: This specifies that the `duration` class template is part of the `std::chrono` namespace, which contains types and functions for time utilities.
- `duration<long long, std::nano>`: The `long long` is the representation type (`Rep`) of the `std::chrono::duration` class template, which is the type used to store the number of ticks (e.g., `int`, `long`, `double`). It indicates that the number of ticks will be stored as a `long long` integer, providing a large range to accommodate very fine-grained durations.
- `std::nano` is the period type (`Period`) of the `std::chrono::duration` class template. The period type represents the tick period (e.g., seconds, milliseconds, nanoseconds). The default is `ratio<1>`, which means the duration is in seconds. `std::ratio` is a template that represents a compile-time rational number. The `std::nano` is a `typedef` for `std::ratio<1, 1000000000>`, which means each tick represents one nanosecond.

The last line is:

```cpp
return duration.count();
```

Where `duration.count()` returns the count of the duration in nanoseconds as a `long long` value, which is the total time taken by `func` to execute.

Whew! That was long and exhausting. I'll try to be more concise in the future. I had to provide some details because most of my students are familiar with Python but have limited knowledge of C++.

The next support function is Function 2, `long long average_time(Func func, int iterations, Args&&... args)`:

```Cpp
// Function 2: to calculate average execution time
template <typename Func, typename... Args>
long long average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    for (int i = 0; i < iterations; ++i) {
        total_time += measure_time(func, std::forward<Args>(args)...);
    }
    return total_time / iterations;
}
```

_Code Fragment 8 - Average Time Function_{: class="legend"}

The `average_time` function template was designed to measure and calculate the average execution time of a given callable entity, such as a function, lambda, or functor, over a specified number of iterations. The template parameters `typename Func` and `typename... Args` allow the function to accept any callable type and a variadic list of arguments that can be forwarded to the callable. The function takes three parameters: the callable entity `func`, the number of iterations `iterations`, and the arguments `args` to be forwarded to the callable. Inside the function, a variable, `total_time`, is initialized to zero to accumulate the total execution time. A loop runs for the specified number of iterations, and during each iteration, the `measure_time` function is called to measure the execution time of `func` with the forwarded arguments, which is then added to `total_time`.

After the loop completes, `total_time` contains the sum of the execution times for all iterations. The function then calculates the average execution time by dividing `total_time` by the number of iterations and returns this value. This approach ensures that the average time provides a more reliable measure of the callable's performance by accounting for variations in execution time across multiple runs. The use of `std::forward<Args>(args)...` in the call to `measure_time` ensures that the arguments are forwarded with their original value categories, maintaining their efficiency and correctness. I like to think that `average_time()` provides a robust method for benchmarking the performance of callable entities in a generic and flexible manner.

I said I would be succinct! Despite all the setbacks, we have reached the
`int main()`:

```Cpp
int main() {

    const int iterations = 1000;
    std::vector<int> test_cases = { 10, 20, 30 }; //fibonacci numbers

    for (int n : test_cases) {
        std::cout << "Calculating Fibonacci(" << n << ")\n";

        // Calculation and average time using the simple recursive function
        long long avg_time_recursive = average_time(fibonacci, iterations, n);
        std::cout << "Average time for recursive Fibonacci: " << avg_time_recursive << " ns\n";

        // Calculation and average time using the memoization function
        std::unordered_map<int, int> memo;
        auto fibonacci_memo_wrapper = [&memo](int n) { return fibonacci_memo(n, memo); };
        long long avg_time_memo = average_time(fibonacci_memo_wrapper, iterations, n);
        std::cout << "Average time for memoized Fibonacci: " << avg_time_memo << " ns\n";

        // Calculation and average time using the tabulation function
        long long avg_time_tabulation = average_time(fibonacci_tabulation, iterations, n);
        std::cout << "Average time for tabulated Fibonacci: " << avg_time_tabulation << " ns\n";

        std::cout << "-----------------------------------\n";
    }

    return 0;
}
```

_Code Fragment 9 - C++ `std::vector` main() function_{: class="legend"}

The `main()` function measures and compares the average execution time of different implementations of the Fibonacci function. Here's a detailed explanation of each part:

The program starts by defining the number of iterations (`const int iterations = 1000;`) and a vector of test cases (`std::vector<int> test_cases = { 10, 20, 30 };`). It then iterates over each test case, calculating the Fibonacci number using different methods and measuring their average execution times.

For the memoized Fibonacci implementation, the program first creates an unordered map `memo` to store previously computed Fibonacci values. It then defines a lambda function `fibonacci_memo_wrapper` that captures `memo` by reference and calls the `fibonacci_memo` function. The `average_time` function is used to measure the average execution time of this memoized implementation over 1000 iterations for each test case.

The other functions follow a similar pattern to measure and print their execution times. For instance, in the case of the recursive Fibonacci function, the line `long long avg_time_recursive = average_time(fibonacci, iterations, n);` calls the `average_time` function to measure the average execution time of the simple recursive Fibonacci function over 1000 iterations for the current test case $n$. The result, stored in `avg_time_recursive`, represents the average time in nanoseconds. The subsequent line, `std::cout << "Average time for recursive Fibonacci: " << avg_time_recursive << " ns\n";`, outputs this average execution time to the console, providing insight into the performance of the recursive method.

The results are printed to the console, showing the performance gain achieved through memoization compared to the recursive and tabulation methods.

#### Running Example 4 - `std::vector`

Example 4, the simple and intuitive code for testing purposes, finds three specific Fibonacci numbers — the 10th, 20th, and 30th — using three different functions, 1,000 times each. This code uses an `int`, `std::vector`, and `std::unordered_map` for storing the values of the Fibonacci sequence and, when executed, presents the following results.

```shell
Calculating Fibonacci(10)
Average time for recursive Fibonacci: 660 ns
Average time for memoized Fibonacci: 607 ns
Average time for tabulated Fibonacci: 910 ns
-----------------------------------
Calculating Fibonacci(20)
Average time for recursive Fibonacci: 75712 ns
Average time for memoized Fibonacci: 444 ns
Average time for tabulated Fibonacci: 1300 ns
-----------------------------------
Calculating Fibonacci(30)
Average time for recursive Fibonacci: 8603451 ns
Average time for memoized Fibonacci: 414 ns
Average time for tabulated Fibonacci: 1189 ns
-----------------------------------
```

_Output 1 - running Example 4 - std::vector_{: class="legend"}

the careful reader should note that the execution times vary non-linearly and, in all cases, for this problem, the Dynamic Programming version using tabulation was faster. There is much discussion about the performance of the Vector class compared to the Array class. To test the performance differences between `std::vector` and `std::array`, we will retry using `std::array`

### Example 5: using `std::array`

First and foremost, `std::array` is a container from the C++ Standard Library with some similarities to, and some differences from, `std::vector`, namely:

> The `std::array` is a template class introduced in C++11, which provides a fixed-size array that is more integrated with the STL than traditional C-style arrays. Unlike `std::vector`, `std::array` does not manage its own memory dynamically; its size is fixed at compile-time, which makes it more efficient for cases where the array size is known in advance and does not change. `std::array` objects can be passed to and returned from functions, and they support copy and assignment operations. They provide the same `begin()`/`end()` methods as vectors, allowing for easy iteration and integration with other STL algorithms. One significant advantage of `std::array` over traditional arrays is that it encapsulates the array size within the type itself, eliminating the need for passing size information separately. Additionally, `std::array` provides member functions such as `size()`, which returns the number of elements in the array, enhancing safety and usability. However, since `std::array` has a fixed size, it does not offer the dynamic resizing capabilities of `std::vector`, making it less flexible in scenarios where the array size might need to change.

When considering performance differences between `std::vector` and `std::array`, it's essential to understand their underlying characteristics and use cases. _`std::array` is a fixed-size array, with its size determined at compile-time, making it highly efficient for situations where the array size is known and constant_. The lack of dynamic memory allocation means that `std::array` avoids the overhead associated with heap allocations, resulting in faster access and manipulation times. This fixed-size nature allows the compiler to optimize memory layout and access patterns, often resulting in better cache utilization and reduced latency compared to dynamically allocated structures.

In contrast, _`std::vector` provides a dynamic array that can grow or shrink in size at runtime, offering greater flexibility but at a cost. The dynamic nature of `std::vector` necessitates managing memory allocations and deallocations, which introduces overhead_. When a `std::vector` needs to resize, it typically allocates a new block of memory and copies existing elements to this new block, an operation that can be costly, especially for large vectors. Despite this, `std::vector` employs strategies such as capacity doubling to minimize the frequency of reallocations, balancing flexibility and performance.

_For small, fixed-size arrays, `std::array` usually outperforms `std::vector` due to its minimal overhead and compile-time size determination_. It is particularly advantageous in performance-critical applications where predictable and low-latency access is required. On the other hand, `std::vector` shines in scenarios where the array size is not known in advance or can change, offering a more convenient and safer alternative to manually managing dynamic arrays.

In summary, `std::array` generally offers superior performance for fixed-size arrays due to its lack of dynamic memory management and the resultant compiler optimizations. However, `std::vector` provides essential flexibility and ease of use for dynamically sized arrays, albeit with some performance trade-offs. The choice between `std::array` and `std::vector` should be guided by the specific requirements of the application, weighing the need for fixed-size efficiency against the benefits of dynamic resizing.

| Feature           | `std::vector`                                    | `std::array`                                   |
| ----------------- | ------------------------------------------------ | ---------------------------------------------- |
| Size              | Dynamic (can change at runtime)                  | Fixed (determined at compile time)             |
| Memory Management | Dynamic allocation on the heap                   | Typically on the stack, no dynamic allocation  |
| Performance       | Can have overhead due to resizing                | Generally more efficient for fixed-size data   |
| Use Cases         | When the number of elements is unknown or varies | When the number of elements is known and fixed |
| Flexibility       | High (can add/remove elements easily)            | Low (size cannot be changed)                   |
| STL Integration   | Yes (works with algorithms and iterators)        | Yes (similar interface to vector)              |

_Tabela 1 - std::vector and std::array comparison_{: class="legend"}

So, we can test this performance advantages, running a code using `std::array`. Since I am lazy, I took the same code used in Example 4 and only replaced the container in the `fibonacci_tabulation` function. You can see it below:

```Cpp
// Iterative function with tabulation to calculate Fibonacci using arrays
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    std::array<int, 41> dp = {};  // array to support up to Fibonacci(40)
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

_Code Fragment 10 - C++, `std::array`, Tabulation Function_{: class="legend"}

This is basically the same code that we discussed in the previous section, only replacing the `std::vector` class with the `std::array` class. Therefore, we do not need to analyze the code line by line and can consider the flowcharts and complexity analysis already performed.

#### Running Example 5: using `std::array`

Running the Example 5 will produces the following result:

```shell
Calculating Fibonacci(10)
Average time for recursive Fibonacci: 807 ns
Average time for memoized Fibonacci: 426 ns
Average time for tabulated Fibonacci: 159 ns
-----------------------------------
Calculating Fibonacci(20)
Average time for recursive Fibonacci: 88721 ns
Average time for memoized Fibonacci: 434 ns
Average time for tabulated Fibonacci: 371 ns
-----------------------------------
Calculating Fibonacci(30)
Average time for recursive Fibonacci: 10059626 ns
Average time for memoized Fibonacci: 414 ns
Average time for tabulated Fibonacci: 439 ns

```

_Output 2 - Example 5 running std::vector_{: class="legend"}

We have reached an interesting point. Just interesting!

We achieved a performance gain using memoization and tabulation, as evidenced by the different complexities among the recursive $O(n^2)$, memoization $O(n)$, and tabulation $O(n)$. Additionally, we observed a slight improvement in execution time by choosing `std::array` instead of `std::vector`. However, we still have some options to explore. Options never end!

### Code 3: C-style Array

We are using a C++ container of integers to store the already calculated Fibonacci numbers as the basis for the two Dynamic Programming processes we are studying so far, memoization and tabulation, one `std::unordered_map` and one `std::vector` or `std::array`. However, there is an even simpler container in C++: the array. The C-Style array.

For compatibility, C++ allows the use of code written in C, including data structures, libraries, and functions. So, why not test these data structures? For this, I wrote new code, keeping the functions using `std::array` and `std::unordered_map` and creating two new dynamic functions using C-style arrays. The code is basically the same except for the following fragment:

```Cpp
const int MAXN = 100;
bool found[MAXN] = { false };
int memo[MAXN] = { 0 };

// New function with memoization using arrays
int cArray_fibonacci_memo(int n) {
    if (found[n]) return memo[n];
    if (n == 0) return 0;
    if (n == 1) return 1;

    found[n] = true;
    return memo[n] = cArray_fibonacci_memo(n - 1) + cArray_fibonacci_memo(n - 2);
}

// New function with tabulation using arrays
int cArray_fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    int dp[MAXN] = { 0 };  // array to support up to MAXN
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

_Code Fragment 11 - C++, C-Style Array, Memoization and Tabulation Functions_{: class="legend"}

As I said, this code segment introduces two new functions for calculating Fibonacci numbers using C-style arrays, with a particular focus on the function for memoization. Instead of using an `std::unordered_map` to store the results of previously computed Fibonacci numbers, the memoization function `cArray_fibonacci_memo` uses two arrays: `found` and `memo`. The `found` array is a boolean array that tracks whether the Fibonacci number for a specific index has already been calculated, while the `memo` array stores the calculated Fibonacci values[^1]. The function checks if the result for the given $n$ is already computed by inspecting the `found` array. If it is, the function returns the value from the `memo` array. If not, it recursively computes the Fibonacci number, stores the result in the `memo` array, and marks the `found` array as true for that index. To be completely honest, this idea of using two arrays comes from [this site](<[URL](https://cp-algorithms.com/dynamic_programming/intro-to-dp.html)>).

The `cArray_fibonacci_tabulation` function, on the other hand, implements the tabulation method using a single C-Style array `dp` to store the Fibonacci numbers up to the $n$th value. The function initializes the base cases for the Fibonacci Sequence, with `dp[0]` set to $0$ and `dp[1]` set to $1$. It then iterates from $2$ to $n$, filling in the `dp` array by summing the two preceding values. This iterative approach avoids the overhead of recursive calls, making it more efficient for larger values of $n$.

Again succinct! I think I'm learning. These structures have the same space and time complexities that we have observed since Example 4. In other words, all that remains is to run this code and evaluate the execution times.

#### Running Code 3: using C-Style Array

Will give us the following answer:

```Shell
Calculating Fibonacci(10)
Average time for recursive Fibonacci: 718 ns
Fibonacci(10) = 55
Average time for memoized Fibonacci: 439 ns
Fibonacci(10) = 55
Average time for tabulated Fibonacci: 67 ns
Fibonacci(10) = 55
Average time for new memoized Fibonacci: 29 ns
Fibonacci(10) = 55
Average time for new tabulated Fibonacci: 72 ns
Fibonacci(10) = 55
-----------------------------------
Calculating Fibonacci(20)
Average time for recursive Fibonacci: 71414 ns
Fibonacci(20) = 6765
Average time for memoized Fibonacci: 449 ns
Fibonacci(20) = 6765
Average time for tabulated Fibonacci: 83 ns
Fibonacci(20) = 6765
Average time for new memoized Fibonacci: 28 ns
Fibonacci(20) = 6765
Average time for new tabulated Fibonacci: 87 ns
Fibonacci(20) = 6765
-----------------------------------
Calculating Fibonacci(30)
Average time for recursive Fibonacci: 8765969 ns
Fibonacci(30) = 832040
Average time for memoized Fibonacci: 521 ns
Fibonacci(30) = 832040
Average time for tabulated Fibonacci: 102 ns
Fibonacci(30) = 832040
Average time for new memoized Fibonacci: 29 ns
Fibonacci(30) = 832040
Average time for new tabulated Fibonacci: 115 ns
Fibonacci(30) = 832040
-----------------------------------
```

_Output 3: running C-Style array_{: class="legend"}

And there it is, we have found a code fast enough for calculating the nth Fibonacci number in an execution time suitable to my ambitions. The only problem is that we used C-Style arrays in a C++ solution. In other words, we gave up all C++ data structures to make the program as fast as possible. We traded a diverse and efficient language for a simple and straightforward one. This choice will be up to the kind reader. You will have to decide if you know enough C to solve any problem or if you need to use predefined data structures to solve your problems. _Unless there is someone in the competitive programming using C. In that case, it's C and that's it_.

Before we start solving problems with Dynamic Programming, let's summarize the execution time reports in a table for easy visualization and to pique the curiosity of the kind reader.

## Execution Time Comparison Table

| Container          | Number | Recursive (ns) | Memoized (ns) | Tabulated (ns) |
| ------------------ | ------ | -------------- | ------------- | -------------- |
| **Vectors**        | 10     | 660            | 607           | 910            |
|                    | 20     | 75,712         | 444           | 1,300          |
|                    | 30     | 8,603,451      | 414           | 1,189          |
| **Arrays**         | 10     | 807            | 426           | 159            |
|                    | 20     | 88,721         | 434           | 371            |
|                    | 30     | 10,059,626     | 414           | 439            |
| **C-Style Arrays** | 10     | 718            | 29            | 72             |
|                    | 20     | 71,414         | 28            | 87             |
|                    | 30     | 8,765,969      | 29            | 115            |

_Tabela 2 - Code Execution Time Comparison_{: class="legend"}

With sufficient practice, Dynamic Programming concepts will become intuitive. I know, the text is dense and complicated. I purposefully mixed concepts of Dynamic Programming, complexity analysis, C++, and performance. If the kind reader is feeling hopeless, stand up, have a soda, walk a bit, and start again. Like everything worthwhile in life, Dynamic Programming requires patience, effort, and time. If, on the other hand, you feel confident, let's move on to our first problem.

## Your First Dynamic Programming Problem

Dynamic Programming concepts became popular in the early 21st century thanks to job interviews for large companies. Until then, only high-performance and competitive programmers were concerned with these techniques. Today, among others, we have [LeetCode](https://leetcode.com/) with hundreds, perhaps thousands of problems to solve. I strongly recommend trying to solve some of them. Here, I will only solve problems whose solutions are already available on other sites. You might even come across some from LeetCode problem, but that will be by accident. The only utility of LeetCode, for me, for you, and for them, is that the problems are not easy to find or solve. Let's start with a problem that is now a classic on the internet and, according to legend, was part of a Google interview.

### The "Two-Sum" problem

**Statement**: In a technical interview, you've been given an array of numbers, and you need to find a pair of numbers that sum up to a given target value. The numbers can be positive, negative, or both. Can you design an algorithm that works in $O(n)$ time complexity or better?

For example, given the array: `[8, 10, 2, 9, 7, 5]` and the target sum: 11

Your function should return a pair of numbers that add up to the target sum. Your answer must be a function in form: `Values(sequence, targetSum)`, In this case, your function should return (9, 2).

#### Brute-Force for Two-Sum's problem

The most obvious solution, usually the first that comes to mind, involves checking all pairs in the array to see if any pair meets the desired target value. This solution is not efficient for large arrays; it has a time complexity of $O(n^2)$ where $n$ is the number of elements in the array. The flow of the Brute-Force function can be seen in Flowchart 4.

![]({{ site.baseurl }}/assets/images/flow4.jpg)
_Flowchart 4 - Brute-Force solution for Two-Sum problem_{: class="legend"}

Flowchart 4 enables the creation of a function to solve the two-sum problem in C++20, as can be seen in Code 4 below:

```cpp
#include <iostream>
#include <vector>
#include <utility>

// Function to find a pair of numbers that add up to the target sum Brute-Force
std::pair<int, int> Values(const std::vector<int>& sequence, int targetSum) {
    int n = sequence.size();

    // Iterate over all possible pairs
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Check if the current pair sums to the target
            if (sequence[i] + sequence[j] == targetSum) {
                return std::make_pair(sequence[i], sequence[j]); // Pair found
            }
        }
    }

    // No pair found
    return std::make_pair(-1, -1);
}

int main() {
    // Example usage
    std::vector<int> sequence = { 8, 10, 2, 9, 7, 5 };
    int targetSum = 11;

    // Call the function and print the result
    std::pair<int, int> result = Values(sequence, targetSum);
    if (result.first != -1) {
        std::cout << "Pair found: (" << result.first << ", " << result.second << ")" << std::endl;
    }
    else {
        std::cout << "No pair found." << std::endl;
    }

    return 0;
}
```

_Code 3: Full code of a two-sum using `std::vector`_{: class="legend"}

The Values function is quite simple, but the use of `std::vector` and `std::pair` in Code 4 deserves a closer look. While `std::array` might offer a slight performance edge, the dynamic nature of `std::vector` makes it a better fit for interview and competitive programming scenarios where the size of the input data isn't known in advance. This flexibility is crucial when dealing with data read from external sources like the terminal or text files.

> `std::pair` is a standard template class in C++ used to store a pair of values, which can be of different types. It has been available since C++98 and is defined in the `<utility>` header. This class is particularly useful for returning two values from a function or for storing two related values together. It has two public member variables, `first` and `second`, which store the values. A `std::pair` can be initialized using constructors or the helper function `std::make_pair`.
> The kind reader can create a `std::pair` directly using its constructor (`std::pair<int, double> p1(42, 3.14);`) or by using `std::make_pair` (`auto p2 = std::make_pair(42, 3.14);`). It is straightforward to access the members `first` and `second` directly (`std::cout << "First: " << p1.first << ", Second: " << p1.second << std::endl;`).
> Since C++11, we also have `std::tuple` and `std::tie`, which extend the functionality of `std::pair` by allowing the grouping of more than two values. `std::tuple` can store any number of values of different types, making it more versatile than `std::pair`. The `std::tie` function can be used to unpack a `std::tuple` into individual variables. While `std::pair` is simpler and more efficient for just two values, `std::tuple` provides greater flexibility for functions needing to return multiple values. For example, a `std::tuple` can be created using `std::make_tuple` (`auto t = std::make_tuple(1, 2.5, "example");`), and its elements can be accessed using `std::get<index>(t)`.

The kind reader may have noticed the use of `(-1, -1)` as sentinel values to indicate that the function did not find any pair. There is a better way to do this. Use the `std::optional` class as we can see in Code 5:

```cpp
#include <iostream>
#include <vector>
#include <optional>
#include <utility>

// Function to find a pair of numbers that add up to the target sum
std::optional<std::pair<int, int>> Values(const std::vector<int>& sequence, int targetSum) {
    int n = sequence.size();

    // Iterate over all possible pairs
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Check if the current pair sums to the target
            if (sequence[i] + sequence[j] == targetSum) {
                return std::make_pair(sequence[i], sequence[j]); // Pair found
            }
        }
    }

    // No pair found
    return std::nullopt;
}

int main() {
    // Example usage
    std::vector<int> sequence = {8, 10, 2, 9, 7, 5};
    int targetSum = 11;

    // Call the function and print the result
    if (auto result = Values(sequence, targetSum)) {
        std::cout << "Pair found: (" << result->first << ", " << result->second << ")" << std::endl;
    } else {
        std::cout << "No pair found." << std::endl;
    }

    return 0;
}
```

_Code 4: Full code of a two-sum using `std::vector` and `std::optional`_{: class="legend"}

> `std::optional` is a feature introduced in C++17 that provides a way to represent optional (or nullable) values. It is a template class that can contain a value or be empty, effectively indicating the presence or absence of a value without resorting to pointers or sentinel values. This makes `std::optional` particularly useful for functions that may not always return a meaningful result. By using `std::optional`, developers can avoid common pitfalls associated with null pointers and special sentinel values, thereby writing safer and more expressive code. `std::optional` is similar to the `Maybe` type in Haskell, which also represents an optional value that can either be `Just` a value or `Nothing`. An equivalent in Python is the use of `None` to represent the absence of a value, often combined with the `Optional` type hint from the `typing` module to indicate that a function can return either a value of a specified type or `None`.

Here is an example of how you can use `Optional` from the `typing` module in Python to represent a function that may or may not return a value:

```python
from typing import Optional

def find_min_max(numbers: list[int]) -> Optional[tuple[int, int]]:
    if not numbers:
        return None

    min_val = min(numbers)
    max_val = max(numbers)

    return min_val, max_val
```

_Code Fragment 12 - Optional implemented in Python_{: class="legend"}

Relying solely on brute-force solutions won't impress interviewers or win coding competitive programmings. It's crucial to strive for solutions with lower time complexity whenever possible. While some problems might not have more efficient alternatives, most interview and competitive programming questions are designed to filter out candidates who only know brute-force approaches.

#### Recursive Approach: Divide and Conquer

The recursive solution leverages a two-pointer approach to efficiently explore the array within a dynamically shrinking window defined by the `start` and `end` indices. It operates by progressively dividing the search space into smaller subproblems, each represented by a narrower window, until a base case is reached or the target sum is found. Here's the refined description, flowchart and code:

#### Base Cases

1. **Empty Input:** If the array is empty (or if the `start` index is greater than or equal to the `end` index), there are no pairs to consider. In this case, we return `std::nullopt` to indicate that no valid pair was found.

2. **Target Sum Found:** If the sum of the elements at the current `start` and `end` indices equals the `target` value, we've found a matching pair. We return this pair as `std::optional<std::pair<int, int>>` to signal success and provide the result.

#### Recursive Step\*\*

1. **Explore Leftward:** We make a recursive call to the function, incrementing the `start` index by one. This effectively shifts our focus to explore pairs that include the next element to the right of the current `start` position.

2. **Explore Rightward (If Necessary):** If the recursive call in step 1 doesn't yield a solution, we make another recursive call, this time decrementing the `end` index by one. This shifts our focus to explore pairs that include the next element to the left of the current `end` position.

This leads us to the illustration of the algorithm in Flowchart 4 and its implementation in C++ Code 5:

![]({{ site.baseurl }}/assets/images/twoSumRecur.jpg)
_Flowchart 4 - Two-Sum problem recursive solution_{: class="legend"}

```Cpp
#include <vector>
#include <optional>
#include <iostream>

// Recursive function to find a pair of numbers that add up to the target sum
std::optional<std::pair<int, int>> findPairRecursively(const std::vector<int>& arr, int target, int start, int end) {
    // Base case: If start index is greater than or equal to end index, no pair is found
    if (start >= end) {
        return std::nullopt; // Return no value (null optional)
    }
    // Base case: If the sum of elements at start and end indices equals the target, pair is found
    if (arr[start] + arr[end] == target) {
        return std::make_optional(std::make_pair(arr[start], arr[end])); // Return the pair
    }
    // Recursive call: Move the start index forward to check the next element
    auto result = findPairRecursively(arr, target, start + 1, end);
    if (result) {
        return result;   // If a pair is found in the recursive call, return it
    }
    // Recursive call: Move the end index backward to check the previous element
    return findPairRecursively(arr, target, start, end - 1);
}

// Function to find a pair of numbers that add up to the target sum
std::optional<std::pair<int, int>> Values(const std::vector<int>& sequence, int targetSum) {
    // Call the recursive function with initial indices (0 and size-1)
    return findPairRecursively(sequence, targetSum, 0, sequence.size() - 1);
}

int main() {
    // Example usage
    std::vector<int> sequence = { 8, 10, 2, 9, 7, 5 }; // Input array
    int targetSum = 11; // Target sum

    // Call the function to find the pair
    auto result = Values(sequence, targetSum);

    // Print the result
    if (result) {
        std::cout << "Pair found: (" << result->first << ", " << result->second << ")\n";
    } else {
        std::cout << "No pair found.\n";
    }
    return 0;
}
```

_Code 5: Full code of a two-sum using a recursive function_{: class="legend"}

#### Solution Analysis

The recursion systematically explores all possible pairs in the array by moving the start and end indices in a controlled manner. With each recursive call, the problem is reduced until one of the base cases is reached.

The `std::optional<std::pair<int, int>> findPairRecursively(const std::vector<int>& arr, int target, int start, int end)` recursive function explores all possible pairs in the array by moving the `start` and `end` indices. Let's analyze its time complexity:

1. **Base Case**: The base case occurs when `start` is greater than or equal to `end`. In the worst case, this happens after exploring all possible pairs.

2. **Recursive Calls**: For each pair `(start, end)`, there are two recursive calls:
   - One that increments the `start` index.
   - Another that decrements the `end` index.

Given an array of size `n`, the total number of pairs to explore is approximately `n^2 / 2` (combinatorial pairs). Since each recursive call reduces the problem size by one element, the number of recursive calls can be modeled as a binary tree with a height of `n`, leading to a total of `2^n` calls in the worst case.

Therefore, the recursive function exhibits a time complexity of $O(2^n)$. This exponential complexity arises because each unique pair of indices in the array triggers recursive calls, leading to a rapid increase in computation time as the array size grows. This makes the recursive approach impractical for handling large arrays.

The space complexity, however, is determined by the maximum depth of the recursion stack:

1. **Recursion Stack Depth**: In the worst-case scenario, the recursive function will delve to a depth of $n$. This happens when each recursive call processes a single element and spawns further recursive calls until it reaches the base case (when a single element remains).

2. **Auxiliary Space**: Besides the space occupied by the recursion stack, no other substantial amount of memory is utilized.

Consequently, the recursive function's space complexity is $O(n)$, where $n$ denotes the array size. This linear space complexity results from the recursion stack, which stores information about each function call.

In summary, we have the following characteristics:

- **Time Complexity**: $O(2^n)$;
- **Space Complexity**: $O(n)$.

While the recursive approach proves to be highly inefficient in terms of time complexity, rendering it unsuitable for large inputs, it's crucial to compare its performance with the previously discussed brute-force solutions to gain a comprehensive understanding of its strengths and weaknesses.

The Brute-Force solution to the two-sum problem involves checking all possible pairs in the array to see if any pair meets the desired target value. This approach has a time complexity of $O(n^2)$ because it uses nested loops to iterate over all pairs. The space complexity is $O(1)$ as it does not require additional storage beyond the input array and a few variables.

On the other hand, the recursive solution systematically explores all possible pairs by moving the `start` and `end` indices. Although it achieves the same goal, its time complexity is much worse, at $O(2^n)$. This exponential complexity arises because each recursive call generates two more calls, leading to an exponential growth in the number of calls. The space complexity of the recursive solution is $O(n)$, as it requires a recursion stack that can grow up to the depth of the array size.

In summary, while both approaches solve the problem, the Brute-Force solution is significantly more efficient in terms of time complexity ($O(n^2)$ vs. $O(2^n)$), and it also has a lower space complexity ($O(1)$ vs. $O(n)$). However, we are not interested in either of these solutions. The Brute-Force solution is naive and offers no advantage, and the recursive solution is impractical. Thus, we are left with the Dynamic Programming solutions.

#### Dynamic Programming: memoization

> Regardless of the efficiency of the recursive code, the first law of Dynamic Programming says: always start with recursion. Thus, the recursive function will be useful for defining the structure of the code using memoization and tabulation.

Memoization is a technique that involves storing the results of expensive function calls and reusing the cached result when the same inputs occur again. By storing intermediate results, we can avoid redundant calculations, thus optimizing the solution.

In the context of the two-sum problem, memoization can help reduce the number of redundant checks by storing the pairs that have already been evaluated. We'll use a `std::unordered_map` to store the pairs of indices we've already checked and their sums. This will help us quickly determine if we've already computed the sum for a particular pair of indices.

We'll modify the recursive function to check the memoization map before performing any further calculations. If the pair has already been computed, we'll use the stored result instead of recalculating. After calculating the sum of a pair, we'll store the result in the memoization map before returning it. This ensures that future calls with the same pair of indices can be resolved quickly. By using memoization, we aim to reduce the number of redundant calculations, thus improving the efficiency compared to a purely recursive approach.

#### Memoized Recursive Solution in C++20

The only significant modification in Code 5 is the conversion of the recursive function to Dynamic Programming with memoization. Code 6 presents this updated function.

```cpp
// Recursive function with memoization to find a pair of numbers that add up to the target sum
std::optional<std::pair<int, int>> findPairRecursivelyMemo(
    const std::vector<int>& arr,
    int target,
    int start,
    int end,
    std::unordered_map<std::string, std::optional<std::pair<int, int>>>& memo
) {
    // Base case: If start index is greater than or equal to end index, no pair is found
    if (start >= end) {
        return std::nullopt; // Return no value (null optional)
    }

    // Create a unique key for memoization
    std::string key = createKey(start, end);

    // Check if the result is already in the memoization map
    if (memo.find(key) != memo.end()) {
        return memo[key]; // Return the memoized result
    }

    // Base case: If the sum of elements at start and end indices equals the target, pair is found
    if (arr[start] + arr[end] == target) {
        auto result = std::make_optional(std::make_pair(arr[start], arr[end]));
        memo[key] = result; // Store the result in the memoization map
        return result; // Return the pair
    }

    // Recursive call: Move the start index forward to check the next element
    auto result = findPairRecursivelyMemo(arr, target, start + 1, end, memo);
    if (result) {
        memo[key] = result; // Store the result in the memoization map
        return result; // If a pair is found in the recursive call, return it
    }

    // Recursive call: Move the end index backward to check the previous element
    result = findPairRecursivelyMemo(arr, target, start, end - 1, memo);
    memo[key] = result; // Store the result in the memoization map
    return result; // Return the result
}
```

_Code Fragment 13 - Two-sum using a Memoized function_{: class="legend"}

#### Complexity Analysis of the Memoized Solution

In the memoized solution, we store the results of the subproblems in a map to avoid redundant calculations. We can analyze the time complexity step-by-step:

1. **Base Case Check**:

   - If the base case is met (when `start >= end`), the function returns immediately. This takes constant time, $O(1)$.

2. **Memoization Check**:

   - Before performing any calculations, the function checks if the result for the current pair of indices (`start`, `end`) is already stored in the memoization map. Accessing the map has an average time complexity of $O(1)$.

3. **Recursive Calls**:
   - The function makes two recursive calls for each pair of indices: one that increments the `start` index and another that decrements the `end` index.
   - In the worst case, without memoization, this would lead to $2^n$ recursive calls due to the binary nature of the recursive calls.

However, with memoization, each unique pair of indices is computed only once and stored. Given that there are $n(n-1)/2$ unique pairs of indices in an array of size $n$, the memoized function will compute the sum for each pair only once. Thus, the total number of unique computations is limited to the number of pairs, which is $O(n^2)$. Therefore, the time complexity of the memoized solution is **O(n^2)**.

The space complexity of the memoized solution is influenced by two primary factors:

1. **Recursion Stack Depth**: In the most extreme scenario, the recursion could reach a depth of $n$. This means $n$ function calls would be active simultaneously, each taking up space on the call stack. This contributes $O(n)$ to the space complexity.

2. **Memoization Map Size**: This map is used to store the results of computations to avoid redundant calculations. The maximum number of unique entries in this map is determined by the number of distinct pairs of indices we can form from a set of $n$ elements. This number is given by the combination formula n^2, which simplifies to $n(n-1)/2$. As each entry in the map takes up constant space, the overall space used by the memoization map is $O(n^2)$.

Combining these two factors, the overall space complexity of the memoized solution is dominated by the larger of the two, which is $O(n^2)$. This leads us to the following conclusions:

- **Time Complexity**: $O(n^2)$ - The time it takes to process the input grows quadratically with the input size due to the nested loops and memoization overhead.
- **Space Complexity**: $O(n^2)$ - The amount of memory required for storing memoized results also increases quadratically with the input size because the memoization table grows proportionally to n^2.

By storing the results of subproblems, the memoized solution reduces redundant calculations, achieving a time complexity of $O(n^2)$. The memoization map and recursion stack together contribute to a space complexity of $O(n^2)$. Although it has the same time complexity as the Brute-Force solution, memoization significantly improves efficiency by avoiding redundant calculations, making it more practical for larger arrays.

The Brute-Force solution involves nested loops to check all possible pairs, leading to a time complexity of $O(n^2)$. This solution does not use any additional space apart from a few variables, so its space complexity is $O(1)$. While straightforward, the Brute-Force approach is not efficient for large arrays due to its quadratic time complexity.

The naive recursive solution, on the other hand, explores all possible pairs without any optimization, resulting in an exponential time complexity of $O(2^n)$. The recursion stack can grow up to a depth of $n$, leading to a space complexity of $O(n)$. This approach is highly inefficient for large inputs because it redundantly checks the same pairs multiple times.

At this point we can create a summary table.

| Solution Type | Time Complexity | Space Complexity |
| ------------- | --------------- | ---------------- |
| Brute-Force   | $O(n^2)$        | $O(1)$           |
| Recursive     | $O(2^n)$        | $O(n)$           |
| Memoized      | $O(n^2)$        | $O(n^2)$         |

_Tabela 3 - Brute-Force, Recursive and Memoized Solutions Complexity Comparison_{: class="legend"}

The situation may seem grim, with the brute-force approach holding the lead as our best solution so far. But don't lose hope just yet! We have a secret weapon up our sleeves: Dynamic Programming with tabulation.

#### Dynamic Programming: tabulation

Think of it like this: we've been wandering through a maze, trying every path to find the treasure (our solution). The brute-force approach means we're checking every single path, even ones we've already explored. It's exhausting and time-consuming.

But Dynamic Programming with tabulation is like leaving breadcrumbs along the way. As we explore the maze, we mark the paths we've already taken. This way, we avoid wasting time revisiting those paths and focus on new possibilities. It's a smarter way to navigate the maze and find the treasure faster.

In the context of our problem, tabulation means creating a table to store solutions to smaller subproblems. As we solve larger problems, we can refer to this table to avoid redundant calculations. It's a clever way to optimize our solution and potentially find the treasure much faster.

So, even though the brute-force approach may seem like the only option right now, don't give up! Attention! Spoiler Alert! With Dynamic Programming and tabulation, we can explore the maze more efficiently and hopefully find the treasure we've been seeking.

#### C++ code for Two-Sum problem using tabulation

The code is:

```Cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <optional>

// Function to find a pair of numbers that add up to the target sum using tabulation
std::optional<std::pair<int, int>> ValuesTabulation(const std::vector<int>& sequence, int targetSum) {
    std::unordered_map<int, int> table; // Hash table to store elements and their indices

    for (int i = 0; i < sequence.size(); ++i) {
        int complement = targetSum - sequence[i];

        // Check if the complement exists in the hash table
        if (table.find(complement) != table.end()) {
            return std::make_optional(std::make_pair(sequence[i], complement)); // Pair found
        }

        // Store the current element in the hash table
        table[sequence[i]] = i;
    }

    // No pair found
    return std::nullopt;
}

int main() {
    // Example usage
    std::vector<int> sequence = {8, 10, 2, 9, 7, 5}; // Input array
    int targetSum = 11; // Target sum

    // Call the function to find the pair
    auto result = ValuesTabulation(sequence, targetSum);

    // Print the result
    if (result) {
        std::cout << "Pair found: (" << result->first << ", " << result->second << ")\n";
    } else {
        std::cout << "No pair found.\n";
    }

    return 0;
}
```

_Code 9: Full code of a two-sum using a tabulated function_{: class="legend"}

The `std::optional<std::pair<int, int>> ValuesTabulation(const std::vector<int>& sequence, int targetSum)` function uses a hash table (`std::unordered_map`) to store elements of the array and their indices. For each element in the array, it calculates the complement, which is the difference between the target sum and the current element. It then checks if the complement exists in the hash table. If the complement is found, a pair that sums to the target has been identified and the function returns this pair. If the complement does not exist, the function stores the current element and its index in the hash table and proceeds to the next element.

#### Complexity Analysis of the Tabulation Function

The `std::optional<std::pair<int, int>> ValuesTabulation(const std::vector<int>& sequence, int targetSum)` function uses a hash table to efficiently find a pair of numbers that add up to the target sum. The function iterates through each element of the array once, making its time complexity $O(n)$. For each element, it calculates the complement (the difference between the target sum and the current element) and checks if this complement exists in the hash table. _Accessing and inserting elements into the hash table both have an average time complexity of $O(1)$, contributing to the overall linear time complexity of the function_.

The space complexity of the function is also $O(n)$, as it uses a hash table to store the elements of the array and their indices. The size of the hash table grows linearly with the number of elements in the array, which is why the space complexity is linear.

Comparing this with the other solutions, the Brute-Force solution has a time complexity of $O(n^2)$ because it involves nested loops to check all possible pairs, and its space complexity is $O(1)$ since it uses only a few additional variables. The recursive solution without optimization has an exponential time complexity of $O(2^n)$ due to redundant calculations in exploring all pairs, with a space complexity of $O(n)$ due to the recursion stack. The memoized solution improves upon the naive recursion by storing results of subproblems, achieving a time complexity of $O(n^2)$ and a space complexity of $O(n^2)$ due to the memoization map and recursion stack.

_In comparison, the tabulation function is significantly more efficient in terms of both time and space complexity. It leverages the hash table to avoid redundant calculations and provides a linear time solution with linear space usage, making it the most efficient among the four approaches._ Wha we can see in the following table.

| Solution Type | Time Complexity | Space Complexity |
| ------------- | --------------- | ---------------- |
| Brute-Force   | $O(n^2)$        | $O(1)$           |
| Recursive     | $O(2^n)$        | $O(n)$           |
| Memoized      | $O(n^2)$        | $O(n^2)$         |
| Tabulation    | $O(n)$          | $O(n)$           |

_Tabela 4 - Brute-Force, Recursive, Memoized and Tabulated Solutions Complexity Comparison_{: class="legend"}

And so, it seems, we have a champion: Dynamic Programming with tabulation! Anyone armed with this technique has a significant advantage when tackling this problem, especially in job interviews where optimization and clever problem-solving are highly valued.

However, let's be realistic: in the fast-paced world of programming competitive programmings, where every millisecond counts, tabulation might not always be the winner. It can require more memory and setup time compared to other approaches, potentially slowing you down in a race against the clock.

So, while tabulation shines in showcasing your understanding of optimization and problem-solving, it's important to be strategic in a competitive programming setting. Sometimes, a simpler, faster solution might be the key to victory, even if it's less elegant.

The bottom line? Mastering Dynamic Programming and tabulation is a valuable asset, but knowing when and where to use it is the mark of a true programming champion. Now, all that's left is to analyze the execution times.

#### Execution Time Analysis

I started by testing with the same code we used to test the Fibonacci functions. However, in my initial analysis, I noticed some inconsistencies in the execution times. To address this, I refined our measurement methodology by eliminating lambda functions and directly measuring execution time within the main loop. This removed potential overhead introduced by the lambdas, leading to more reliable results. So, I wrote a new, simpler, and more direct code to test the functions:

```Cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <optional>
#include <utility>
#include <chrono>

// Function to measure execution time
template <typename Func, typename... Args>
long long measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return duration.count();
}

// Function to calculate average execution time
template <typename Func, typename... Args>
long long average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    for (int i = 0; i < iterations; ++i) {
        total_time += measure_time(func, std::forward<Args>(args)...);
    }
    return total_time / iterations;
}

// Brute-Force Solution
std::pair<int, int> ValuesBruteForce(const std::vector<int>& sequence, int targetSum) {
    int n = sequence.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (sequence[i] + sequence[j] == targetSum) {
                return std::make_pair(sequence[i], sequence[j]);
            }
        }
    }
    return std::make_pair(-1, -1);
}

// Naive Recursive Solution
std::optional<std::pair<int, int>> findPairRecursively(const std::vector<int>& arr, int target, int start, int end) {
    if (start >= end) {
        return std::nullopt;
    }
    if (arr[start] + arr[end] == target) {
        return std::make_optional(std::make_pair(arr[start], arr[end]));
    }
    auto result = findPairRecursively(arr, target, start + 1, end);
    if (result) {
        return result;
    }
    return findPairRecursively(arr, target, start, end - 1);
}

std::optional<std::pair<int, int>> ValuesRecursive(const std::vector<int>& sequence, int targetSum) {
    return findPairRecursively(sequence, targetSum, 0, sequence.size() - 1);
}

// Memoized Recursive Solution
std::string createKey(int start, int end) {
    return std::to_string(start) + "," + std::to_string(end);
}

std::optional<std::pair<int, int>> findPairRecursivelyMemo(
    const std::vector<int>& arr, int target, int start, int end,
    std::unordered_map<std::string, std::optional<std::pair<int, int>>>& memo) {
    if (start >= end) {
        return std::nullopt;
    }
    std::string key = createKey(start, end);
    if (memo.find(key) != memo.end()) {
        return memo[key];
    }
    if (arr[start] + arr[end] == target) {
        auto result = std::make_optional(std::make_pair(arr[start], arr[end]));
        memo[key] = result;
        return result;
    }
    auto result = findPairRecursivelyMemo(arr, target, start + 1, end, memo);
    if (result) {
        memo[key] = result;
        return result;
    }
    result = findPairRecursivelyMemo(arr, target, start, end - 1, memo);
    memo[key] = result;
    return result;
}

std::optional<std::pair<int, int>> ValuesMemoized(const std::vector<int>& sequence, int targetSum) {
    std::unordered_map<std::string, std::optional<std::pair<int, int>>> memo;
    return findPairRecursivelyMemo(sequence, targetSum, 0, sequence.size() - 1, memo);
}

// Tabulation Solution
std::optional<std::pair<int, int>> ValuesTabulation(const std::vector<int>& sequence, int targetSum) {
    std::unordered_map<int, int> table;
    for (int i = 0; i < sequence.size(); ++i) {
        int complement = targetSum - sequence[i];
        if (table.find(complement) != table.end()) {
            return std::make_optional(std::make_pair(sequence[i], complement));
        }
        table[sequence[i]] = i;
    }
    return std::nullopt;
}

int main() {
    std::vector<int> sequence = {8, 10, 2, 9, 7, 5};; // 40 numbers
    int targetSum = 11;
    int iterations = 1000;

    std::cout << "-----------------------------------\n";
    std::cout << "Calculating Two-Sum (" << targetSum << ")\n";

    // Measure average execution time for Brute-Force Solution
    auto bruteForceTime = average_time([](const std::vector<int>& seq, int target) {
        ValuesBruteForce(seq, target);
        }, iterations, sequence, targetSum);
    std::cout << "Average time for Brute-Force: " << bruteForceTime << " ns\n";

    // Measure average execution time for Naive Recursive Solution
    auto recursiveTime = average_time([](const std::vector<int>& seq, int target) {
        ValuesRecursive(seq, target);
        }, iterations, sequence, targetSum);
    std::cout << "Average time for Recursive: " << recursiveTime << " ns\n";

    // Measure average execution time for Memoized Recursive Solution
    auto memoizedTime = average_time([](const std::vector<int>& seq, int target) {
        ValuesMemoized(seq, target);
        }, iterations, sequence, targetSum);
    std::cout << "Average time for Memoized: " << memoizedTime << " ns\n";

    // Measure average execution time for Tabulation Solution
    auto tabulationTime = average_time([](const std::vector<int>& seq, int target) {
        ValuesTabulation(seq, target);
        }, iterations, sequence, targetSum);
    std::cout << "Average time for Tabulation: " << tabulationTime << " ns\n";

    std::cout << "-----------------------------------\n";

    return 0;
}
```

_Code 10: Code for execution time test of all functions we create to Two-Sum problem._{: class="legend"}

I simply replicated the functions from the previous code snippets, without any optimization, precisely because our current objective is to solely examine the execution times. Running the new code, we have the following output:

```Shell
-----------------------------------
Calculating Two-Sum (18)
Average time for Brute-Force: 217 ns
Average time for Recursive: 415 ns
Average time for Memoized: 41758 ns
Average time for Tabulation: 15144 ns
-----------------------------------
```

_Output 4: Execution time of Two-Sum solutions._{: class="legend"}

As we've seen, when dealing with a small amount of input data, the brute-force approach surprisingly outshines even more complex algorithms. This might seem counterintuitive, but it's all about the hidden costs of memory management.

When we use sophisticated data structures like `std::string` and `std::unordered_map`, we pay a price in terms of computational overhead. Allocating and deallocating memory on the heap for these structures takes time and resources. This overhead becomes especially noticeable when dealing with small datasets, where the time spent managing memory can easily overshadow the actual computation involved. On the other hand, the brute-force method often relies on simple data types and avoids dynamic memory allocation, resulting in a faster and more efficient solution for smaller inputs.

#### The Dynamic Memory Bottleneck

There are some well-known bottlenecks that can explain why a code with lower complexity runs much slower in a particular environment.

**Hash Table Overhead**: Both memoized and tabulation solutions rely on `std::unordered_map`, which inherently involves dynamic memory allocation. Operations like insertions and lookups, while powerful, come with a cost due to memory management overhead. This is typically slower than accessing elements in a simple array.

**Recursion's Toll**: The naive recursive and memoized solutions utilize deep recursion, leading to a considerable overhead from managing the recursion stack. Each recursive call adds a new frame to the stack, requiring additional memory operations that accumulate over time.

**Memoization's Complexity**: While memoization optimizes by storing intermediate results, it also introduces complexity through the use of `std::unordered_map`. Each new pair calculated requires storage in the hash table, involving dynamic allocations and hash computations, adding to the overall time complexity.

**Cache Friendliness**: Dynamic memory allocations often lead to suboptimal cache utilization. In contrast, the Brute-Force and tabulation solutions likely benefit from better cache locality due to their predominant use of array accesses. Accessing contiguous memory locations (as in arrays) generally results in faster execution due to improved cache hit rates.

**Function Call Overhead**: The overhead from frequent function calls, including those made through lambda functions, can accumulate, particularly in performance-critical code.

By understanding and mitigating these bottlenecks, we can better optimize our code and achieve the expected performance improvements.

_In essence, the dynamic nature of memory operations associated with hash tables and recursion significantly affects execution times_. These operations are generally slower than accessing static memory structures like arrays. The deep recursion in the memoized and naive recursive approaches exacerbates this issue, as the growing recursion stack necessitates increased memory management.

The memoized solution, while clever, bears the brunt of both issues – extensive recursion and frequent hash table operations. This combination leads to higher execution times compared to the Brute-Force and tabulation approaches, which primarily rely on array accesses and enjoy the benefits of better cache performance and reduced memory management overhead.

In conclusion, the observed differences in execution times can be attributed to the distinct memory access patterns and associated overheads inherent in each approach. Understanding these nuances is crucial for making informed decisions when optimizing code for performance.

### We will always have C

As we delve into Dynamic Programming with C++, our focus is on techniques that shine in interviews and coding competitive programmings. Since competitive coding often favors slick C-style code, we'll zero in on a tabulation solution for this problem. Tabulation, as we know, is usually the most efficient approach. To show you what I mean, check out the `int* ValuesTabulationCStyle(const int* sequence, int length, int targetSum)` function in Code Fragment 12.

```Cpp
int* ValuesTabulationCStyle(const int* sequence, int length, int targetSum) {
    const int MAX_VAL = 1000; // Assuming the values in the sequence are less than 1000
    static int result[2] = { -1, -1 }; // Static array to return the result
    int table[MAX_VAL];
    memset(table, -1, sizeof(table));

    for (int i = 0; i < length; ++i) {
        int complement = targetSum - sequence[i];
        if (complement >= 0 && table[complement] != -1) {
            result[0] = sequence[i];
            result[1] = complement;
            return result;
        }
        table[sequence[i]] = i;
    }
    return result;
}
```

_Code Fragment 14 - Two-sum with C-Style code, using a Memoized Function_{: class="legend"}

The C-style function is as straightforward as it gets, and as far as I can tell, it's equivalent to the C++ tabulation function. Perhaps the only thing worth noting is the use of the memset function.

> The `memset` function in C and C++ is your go-to tool for filling a block of memory with a specific value. You'll usually find it defined in the `<cstring>` header. It takes a pointer to the memory block you want to fill (`ptr`), the value you want to set (converted to an `unsigned char`), and the number of bytes you want to set.
> In our code, `memset(table, -1, sizeof(table))`; does exactly that—it fills the entire table array with the value $-1$, which is a handy way to mark those spots as empty or unused.
> Why use `memset` instead of `malloc` or `alloc`? Well, `memset` is all about initializing memory that's already been allocated. `malloc` and `alloc` are for allocating new memory. If you need to do both, C++ has you covered with the `new` operator, which can allocate and initialize in one step.
> So, `memset` is the memory magician for resetting or initializing arrays and structures, while `malloc`, `alloc`, and `calloc` handle the memory allocation part of the job.

The use of `menset` bring us to analise the function complexity.

#### Two-Sum C-Style Tabulation Function Complexity

The function `ValuesTabulationCStyle` uses `memset` to initialize the `table` array. The complexity of the function can be divided into two parts:

1. **Initialization with `memset`:**

   ```cpp
   memset(table, -1, sizeof(table));
   ```

   The memset function initializes each byte of the array. Since the size of the array is constant (`MAX_VAL`), the complexity of this operation is $O(1)$ in terms of asymptotic complexity, although it has a constant cost depending on the array size.

   ```cpp
   for (int i = 0; i < length; ++i) {
       int complement = targetSum - sequence[i];
       if (complement >= 0 && table[complement] != -1) {
           result[0] = sequence[i];
           result[1] = complement;
           return result;
       }
       table[sequence[i]] = i;
   }
   ```

2. The Main Loop

   The main loop iterates over each element of the input array sequence. Therefore, the complexity of this part is $O(n)$, where n is the length of the sequence array.

Combining all the pieces, the function's overall complexity is still $O(n)$. Even though initializing the array with `memset` takes $O(1)$ time, it doesn't change the fact that we have to loop through each element in the input array, which takes $O(n)$ time.

Looking back at our original C++ function using `std::unordered_map`, we also had `O(1)` complexity for the map initialization and `O(n)` for the main loop. So, while using `memset` for our C-style array feels different, it doesn't change the big picture – both approaches end up with linear complexity.

The key takeaway here is that while using `memset` might feel like a win for initialization time, it doesn't change the overall complexity when you consider the whole function. This leaves us with the task of running the code and analyzing the execution times, as shown in Output 5.

```Shell
-----------------------------------
Calculating Two-Sum (11)
Average time for Brute-Force: 318 ns
Average time for Recursive: 626 ns
Average time for Memoized: 39078 ns
Average time for Tabulation: 5882 ns
Average time for Tabulation C-Style: 189 ns
-----------------------------------
```

_Output 5: Execution time of Two-Sum solutions, including C-Style Arrays._{: class="legend"}

Analyzing Output 5, it's clear that the C-style solution is, for all intents and purposes, twice as fast as the C++ tabulation solution. However, there are a few caveats: the C++ code was written to showcase the language's flexibility, not to optimize for performance. On the other hand, the C-style function was designed for simplicity. Often, simplicity equates to speed, and this is something to maximize when creating a function with linear complexity. _Now, we need to compare the C solution with C++ code that prioritizes high performance over flexibility in writing_.

### High Performance C++

Code Fragment 15 was rewritten by stripping away all the complex data structures we were previously using. The `main()` function remains largely unchanged, so it's been omitted here. I've also removed the functions used for measuring and printing execution times.

```Cpp
// Brute-Force Solution
std::array<int, 2> ValuesBruteForce(const std::vector<int>& sequence, int targetSum) {
    int n = sequence.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (sequence[i] + sequence[j] == targetSum) {
                return { sequence[i], sequence[j] };
            }
        }
    }
    return { -1, -1 };
}

// Naive Recursive Solution
std::array<int, 2> findPairRecursively(const std::vector<int>& arr, int target, int start, int end) {
    if (start >= end) {
        return { -1, -1 };
    }
    if (arr[start] + arr[end] == target) {
        return { arr[start], arr[end] };
    }
    std::array<int, 2> result = findPairRecursively(arr, target, start + 1, end);
    if (result[0] != -1) {
        return result;
    }
    return findPairRecursively(arr, target, start, end - 1);
}

std::array<int, 2> ValuesRecursive(const std::vector<int>& sequence, int targetSum) {
    return findPairRecursively(sequence, targetSum, 0, sequence.size() - 1);
}

// Memoized Recursive Solution
std::array<int, 2> findPairRecursivelyMemo(
    const std::vector<int>& arr, int target, int start, int end,
    std::array<std::array<int, 2>, 1000>& memo) {
    if (start >= end) {
        return { -1, -1 };
    }
    if (memo[start][end][0] != -1) {
        return memo[start][end];
    }
    if (arr[start] + arr[end] == target) {
        memo[start][end] = { arr[start], arr[end] };
        return { arr[start], arr[end] };
    }
    std::array<int, 2> result = findPairRecursivelyMemo(arr, target, start + 1, end, memo);
    if (result[0] != -1) {
        memo[start][end] = result;
        return result;
    }
    result = findPairRecursivelyMemo(arr, target, start, end - 1, memo);
    memo[start][end] = result;
    return result;
}

std::array<int, 2> ValuesMemoized(const std::vector<int>& sequence, int targetSum) {
    std::array<std::array<int, 2>, 1000> memo;
    for (auto& row : memo) {
        row = { -1, -1 };
    }
    return findPairRecursivelyMemo(sequence, targetSum, 0, sequence.size() - 1, memo);
}

// Tabulation Solution using C-style arrays
std::array<int, 2> ValuesTabulationCStyle(const int* sequence, int length, int targetSum) {
    const int MAX_VAL = 1000; // Assuming the values in the sequence are less than 1000
    std::array<int, 2> result = { -1, -1 }; // Static array to return the result
    int table[MAX_VAL];
    memset(table, -1, sizeof(table));

    for (int i = 0; i < length; ++i) {
        int complement = targetSum - sequence[i];
        if (complement >= 0 && table[complement] != -1) {
            result[0] = sequence[i];
            result[1] = complement;
            return result;
        }
        table[sequence[i]] = i;
    }
    return result;
}
```

_Code Fragment 15 - All Two-sum functions including a pure `std::array` tabulated function_{: class="legend"}

Running this modified code, we get the following output:

```Shell
-----------------------------------
Calculating Two-Sum (11)
Average time for Brute-Force: 157 ns
Average time for Recursive: 652 ns
Average time for Memoized: 39514 ns
Average time for Tabulation: 5884 ns
Average time for Tabulation C-Style: 149 ns
-----------------------------------
```

_Output 6: Execution time of All Two-Sum solutions including C-style Tabulation._{: class="legend"}

Let's break down the results for calculating the Two-Sum problem, keeping in mind that the fastest solutions have a linear time complexity, O(n):

- **Brute-Force**: Blazing fast at 157 ns on average. This is our baseline, but remember, Brute-Force doesn't always scale well for larger problems.
- **Recursive**: A bit slower at 652 ns. Recursion can be elegant, but it can also lead to overhead.
- **Memoized**: This one's the outlier at 39514 ns. Memoization can be a powerful optimization, but it looks like the overhead is outweighing the benefits in this case.
- **Tabulation**: A respectable 5884 ns. Tabulation is a solid Dynamic Programming technique, and it shows here.
- **Tabulation C-Style**: A close winner at 149 ns! This stripped-down, C-style implementation of tabulation is just a hair behind Brute-Force in terms of speed.

The C++ and C versions of our tabulation function are practically neck and neck in terms of speed for a few key reasons:

1. **Shared Simplicity**: Both versions use plain old arrays (`std::array` in C++, C-style arrays in C) to store data. This means memory access is super efficient in both cases.

2. **Speedy Setup**: We use `memset` to initialize the array in both versions. This is a highly optimized operation, so the initialization step is lightning fast in both languages.

3. **Compiler Magic**: Modern C++ compilers are incredibly smart. They can often optimize the code using `std::array` to be just as fast, or even faster, than hand-written C code.

4. **No Frills**: Both functions have a straightforward design without any fancy branching or extra layers. The operations within the loops are simple and take the same amount of time every time, minimizing any overhead.

5. **Compiler Boost**: _Compilers like GCC and Clang have a whole bag of tricks to make code run faster, like loop unrolling and smart prediction_. These tricks work equally well for both C and C++ code, especially when we're using basic data structures and simple algorithms.

Thanks to all this, the C++ version of our function, using `std::array`, runs just as fast as its C counterpart.

And this is how C++ code should be for competitive programmings. However, not for interviews. In interviews, unless high performance is specifically requested, what they're looking for is your mastery of the language and the most efficient algorithms. So, an O(n) solution using the appropriate data structures will give you a better chance of success.

### Exercises: Variations of the Two Sum

There are few interesting variations of Two-Sum problem:

1. The array can contain both positive and negative integers.
2. Each input would have exactly one solution, and you may not use the same element twice.
3. Each input can have multiple solutions, and the same element cannot be used twice in a pair.
4. The function should return all pairs that sum to the target value.

Try to solve these variations. Take as much time as you need; I will wait.

## The Dynamic Programming Classic Problems

From now on, we will explore 10 classic Dynamic Programming problems. For each one, we will delve into Brute-Force techniques, recursion, memoization, tabulation, and finally the most popular solution for each, even if it is not among the techniques we have chosen. The problems we will address are listed in the table below[^2].

| Name                                           | Description/Example                                                                                                                                                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Counting all possible paths in a matrix        | Given $N$ and $M$, count all possible distinct paths from $(1,1)$ to $(N, M)$, where each step is either from $(i,j)$ to $(i+1,j)$ or $(i,j+1)$.                                                                               |
| Subset Sum                                     | Given $N$ integers and $T$, determine whether there exists a subset of the given set whose elements sum up to $T$.                                                                                                             |
| Longest Increasing Subsequence                 | You are given an array containing $N$ integers. Your task is to determine the LCS in the array, i.e., LCS where every element is larger than the previous one.                                                                 |
| Rod Cutting                                    | Given a rod of length $n$ units, Given an integer array cuts where cuts[i] denotes a position you should perform a cut at. The cost of one cut is the length of the rod to be cut. What is the minimum total cost of the cuts. |
| Longest Common Subsequence                     | You are given strings $s$ and $t$. Find the length of the longest string that is a subsequence of both $s$ and $t$.                                                                                                            |
| Longest Palindromic Subsequence                | Finding the Longest Palindromic Subsequence (LPS) of a given string.                                                                                                                                                           |
| Edit Distance                                  | The edit distance between two strings is the minimum number of operations required to transform one string into the other. Operations are ["Add", "Remove", "Replace"].                                                        |
| Coin Change Problem                            | Given an array of coin denominations and a target amount, find the minimum number of coins needed to make up that amount.                                                                                                      |
| 0-1 knapsack                                   | Given $W$, $N$, and $N$ items with weights $w_i$ and values $v_i$, what is the maximum $\sum_{i=1}^{k} v_i$ for each subset of items of size $k$ ($1 \le k \le N$) while ensuring $\sum_{i=1}^{k} w_i \le W$?                  |
| Longest Path in a Directed Acyclic Graph (DAG) | Finding the longest path in Directed Acyclic Graph (DAG).                                                                                                                                                                      |
| Traveling Salesman Problem (TSP)               | Given a list of cities and the distances between each pair of cities, find the shortest possible route that visits each city exactly once and returns to the origin city.                                                      |
| Matrix Chain Multiplication                    | Given a sequence of matrices, find the most efficient way to multiply these matrices together. The problem is not actually to perform the multiplications, but merely to decide in which order to perform the multiplications. |

_Tabela 5 - The Dynamic Programming we'll study and solve._{: class="legend"}

Stop for a moment, perhaps have a soda or a good wine. Rest a bit, then gather your courage, arm yourself with patience, and continue. Practice makes perfect.

This will continue!!!

## Problem 1 Statement: Counting All Possible Paths in a Matrix

Given two integers $m$ and $n$, representing the dimensions of a matrix, count all possible distinct paths from the top-left corner $(0,0)$ to the bottom-right corner $(m-1,n-1)$. Each step can either be to the right or down.

### Input

- Two integers $m$ and $n$ where $1 \leq m, n \leq 100$.

### Output

- An integer representing the number of distinct paths from $(0,0)$ to $(m-1,n-1)$.

### Example

Input:
3 3

Output:
6

### Constraints

- You can only move to the right or down in each step.

### Analysis

Let's delve deeper into the "unique paths" problem. Picture a matrix as a grid, where you start at the top-left corner $(0, 0)$ and your goal is to reach the bottom-right corner $(m-1, n-1)$. The twist? You're only allowed to move down or right. The challenge is to figure out how many distinct paths you can take to get from start to finish.

This problem so intriguing because it blends the elegance of combinatorics (the study of counting) with the power of Dynamic Programming (a clever problem-solving technique). The solution involves combinations, a fundamental concept in combinatorics. Moreover, it's a classic example of how Dynamic Programming can streamline a seemingly complex problem.

The applications of this type of problem extend beyond theoretical interest. They can be found in practical scenarios like robot navigation in a grid environment, calculating probabilities in games with grid-like movements, and analyzing maze-like structures. Understanding this problem provides valuable insights into a range of fields, from mathematics to robotics and beyond.

Now, let's bring in some combinatorics! To journey from the starting point $(0, 0)$ to the destination $(m-1, n-1)$, you'll need a total of $(m - 1) + (n - 1)$ moves. That's a mix of downward and rightward steps.

The exciting part is figuring out how many ways we can arrange those moves. Imagine choosing $(m - 1)$ moves to go down and $(n - 1)$ moves to go right, out of a total of $(m + n - 2)$ moves. This can be calculated using the following formula:

$$C(m + n - 2, m - 1) = \frac{(m + n - 2)!}{(m - 1)! * (n - 1)!}$$

For our 3x3 matrix example $(m = 3, n = 3)$, the calculation is:

$$C(3 + 3 - 2, 3 - 1) = \frac{4!}{(2! * 2!)} = 6$$

This tells us there are $6$ distinct paths to reach the bottom-right corner. Let's also visualize this using Dynamic Programming:

**Filling the Matrix with Dynamic Programming:**

1. **Initialize:** Start with a $dp$ matrix where $dp[0][0] = 1$ (one way to be at the start).

   dp = \begin{bmatrix}
   1 & 0 & 0 \
   0 & 0 & 0 \
   0 & 0 & 0
   \end{bmatrix}

2. **Fill First Row and Column:** There's only one way to reach each cell in the first row and column (either from the left or above).

   dp = \begin{bmatrix}
   1 & 1 & 1 \
   1 & 0 & 0 \
   1 & 0 & 0
   \end{bmatrix}

3. **Fill Remaining Cells:** For the rest, the number of paths to a cell is the sum of paths to the cell above and the cell to the left: $dp[i][j] = dp[i-1][j] + dp[i][j-1]$

   dp = \begin{bmatrix}
   1 & 1 & 1 \
   1 & 2 & 3 \
   1 & 3 & 6
   \end{bmatrix}

The bottom-right corner, $dp[2][2]$, holds our answer: $6$ unique paths.

Bear with me, dear reader, as I temporarily diverge from our exploration of Dynamic Programming. Before delving deeper, it's essential to examine how we might solve this problem using a Brute-Force approach.

### Using Brute-Force

To tackle the unique paths problem with a Brute-Force approach, we can use an iterative solution and a stack in C++20. The stack will keep track of our current position in the matrix and the number of paths that led us there. Here's a breakdown of how it works:

1. We'll define a structure called `Position` to store the current coordinates $(i, j)$ and the count of paths leading to that position.

2. We'll start by pushing the initial position $(0, 0)$ onto the stack, along with a path count of $1$ (since there's one way to reach the starting point).

3. While the stack isn't empty:

   - Pop the top position from the stack.
   - If we've reached the bottom-right corner $(m-1, n-1)$, increment the total path count.
   - If moving right is possible (within the matrix bounds), push the new position (with the same path count) onto the stack.
   - If moving down is possible, do the same.

4. When the stack is empty, the total path count will be the answer we seek – the total number of unique paths from $(0, 0)$ to $(m-1, n-1)$.

Code Fragment 16 demonstrates how to implement this algorithm in C++.

```cpp
#include <iostream>
#include <stack>
#include <chrono>

//.....
// Structure to represent a position in the matrix
struct Position {
    int i, j;
    int pathCount;
};

// Function to count paths using Brute-Force
int countPaths(int m, int n) {
    std::stack<Position> stk;
    stk.push({ 0, 0, 1 });
    int totalPaths = 0;

    while (!stk.empty()) {
        Position pos = stk.top();
        stk.pop();

        int i = pos.i, j = pos.j, pathCount = pos.pathCount;

        // If we reach the bottom-right corner, add to total paths
        if (i == m - 1 && j == n - 1) {
            totalPaths += pathCount;
            continue;
        }

        // Move right if within bounds
        if (j + 1 < n) {
            stk.push({ i, j + 1, pathCount });
        }

        // Move down if within bounds
        if (i + 1 < m) {
            stk.push({ i + 1, j, pathCount });
        }
    }

    return totalPaths;
}
```

_Code Fragment 16 - Count all paths function using Brute-Force._{: class="legend"}

Let's start looking at `std::stack` data structure:

> In C++20, the `std::stack` is a part of the Standard Template Library (STL) and is used to implement a stack data structure, which follows the Last-In-First-Out (LIFO) principle. A `std::stack` is a container adapter that provides a stack interface, designed to operate in a LIFO context. Elements are added to and removed from the top of the stack. The syntax for creating a stack is `std::stack<T> stack_name;` where `T` is the type of elements contained in the stack.
> To create a stack, you can use constructors such as `std::stack<int> s1;` for the default constructor or `std::stack<int> s2(some_container);` to construct with a container. To add an element to the top of the stack, use the `push` function: `stack_name.push(value);`. To remove the element at the top of the stack, use the `pop` function: `stack_name.pop();`. To return a reference to the top element of the stack, use `T& top_element = stack_name.top();`. To check whether the stack is empty, use `bool is_empty = stack_name.empty();`. Finally, to return the number of elements in the stack, use `std::size_t stack_size = stack_name.size();`.
> The `std::stack` class is a simple and effective way to manage a stack of elements, ensuring efficient access and modification of the last inserted element.

In the context of our code for counting paths in a matrix using a stack, the line `stk.push({ 0, 0, 1 });` is used to initialize the stack with the starting position of the matrix traversal. The `stk` is a `std::stack` of `Position` structures. The `Position` structure is defined as follows:

```cpp
struct Position {
    int i, j;
    int pathCount;
};
```

The `Position` structure has three members: `i` and `j`, which represent the current coordinates in the matrix, and `pathCount`, which represents the number of paths that lead to this position.

The line `stk.push({ 0, 0, 1 });` serves a specific purpose in our algorithm:

1. It creates a `Position` object with the initializer list `{ 0, 0, 1 }`, setting `i = 0`, `j = 0`, and `pathCount = 1`.
2. It then pushes this `Position` object onto the stack using the `push` function of `std::stack`.

In simpler terms, this line of code initializes the stack with the starting position of the matrix traversal, which is the top-left corner of the matrix at coordinates $(0, 0)$. It also sets the initial path count to $1$, indicating that there is one path starting from this position. From here, the algorithm will explore all possible paths from the top-left corner.

_The Brute-Force solution for counting paths in a matrix involves exploring all possible paths from the top-left corner to the bottom-right corner. Each path consists of a series of moves either to the right or down. The algorithm uses a stack to simulate the recursive exploration of these paths_.

To analyze the time complexity, consider that the function must explore each possible combination of moves in the matrix. For a matrix of size $m \times n$, there are a total of $m + n - 2$ moves required to reach the bottom-right corner from the top-left corner. Out of these moves, $m - 1$ must be down moves and $n - 1$ must be right moves. The total number of distinct paths is given by the binomial coefficient $C(m+n-2, m-1)$, which represents the number of ways to choose $m-1$ moves out of $m+n-2$.

The time complexity of the Brute-Force approach is exponential in nature because it explores every possible path. Specifically, the time complexity is $O(2^{m+n})$ since each step involves a choice between moving right or moving down, leading to $2^{m+n}$ possible combinations of moves in the worst case. This exponential growth makes the Brute-Force approach infeasible for large matrices, as the number of paths grows very quickly with increasing $m$ and $n$.

The space complexity is also significant because the algorithm uses a stack to store the state of each position it explores. In the worst case, the depth of the stack can be as large as the total number of moves, which is $m + n - 2$. Thus, the space complexity is $O(m+n)$, primarily due to the stack's storage requirements.

In summary, the Brute-Force solution has an exponential time complexity $O(2^{m+n})$ and a linear space complexity $O(m+n)$, making it suitable only for small matrices where the number of possible paths is manageable.

My gut feeling tells me this complexity is very, very bad. We'll definitely find better complexities as we explore Dynamic Programming solutions. Either way, we need to measure the runtime. Running the function `int countPaths(int m, int n)` within the same structure we created earlier to measure execution time, we will have:

```Shell
-----------------------------------
Calculating Paths in a 3x3 matrix
Average time for Brute-Force: 10865 ns
-----------------------------------
```

_Output 7: Execution time of Counting all paths problem using Brute-Force._{: class="legend"}

Finally, I won't be presenting the code done with pure recursion. As we've seen, recursion is very elegant and can score points in interviews. However, the memoization solution will include recursion, so if you use memoization and recursion in the same solution, you'll ace the interview.

### Using Memoization

Code Fragment 17 shows the functions I created to apply memoization. There are two functions: the `int countPathsMemoizationWrapper(int m, int n)` function used to initialize the dp data structure and call the recursive function `int countPathsMemoization(int m, int n, std::vector<std::vector<int>>& dp)`. I used `std::vector` already anticipating that we won't know the size of the matrix beforehand.

```Cpp
// Function to count paths using Dynamic Programming with memoization
int countPathsMemoization(int m, int n, std::vector<std::vector<int>>& dp) {
    if (m == 1 || n == 1) return 1;  // Base case
    if (dp[m - 1][n - 1] != -1) return dp[m - 1][n - 1];  // Return memoized result
    dp[m - 1][n - 1] = countPathsMemoization(m - 1, n, dp) + countPathsMemoization(m, n - 1, dp);  // Memoize result
    return dp[m - 1][n - 1];
}

int countPathsMemoizationWrapper(int m, int n) {
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, -1));
    return countPathsMemoization(m, n, dp);
}

```

_Code Fragment 17 - Count all paths function using Memoization._{: class="legend"}

The function `int countPathsMemoization(int m, int n, std::vector<std::vector<int>>& dp)` counts all possible paths in an $m \times n$ matrix using Dynamic Programming with memoization. The `dp` matrix serves as a cache, storing and reusing intermediate results to avoid redundant calculations.

Upon invocation, the function first checks if the current position is in the first row (`m == 1`) or first column (`n == 1`). If so, it returns $1$, as there is only one way to reach such cells. Next, it checks the `dp[m-1][n-1]` cell. If the value is not $-1$ (the initial value indicating "not yet calculated"), it signifies that the result for this position has already been memoized and is immediately returned.

Otherwise, the function recursively calculates the number of paths from the cell above (`m-1`, `n`) and the cell to the left (`m`, `n-1`). The sum of these values is then stored in `dp[m-1][n-1]` before being returned.

Utilizing `std::vector<std::vector<int>>` for `dp` ensures efficient storage of intermediate results, significantly reducing the number of recursive calls required compared to a purely recursive approach. This memoized version substantially improves the function's performance, especially for larger matrices.

The `countPathsMemoization` function exemplifies memoization rather than tabulation due to its top-down recursive approach, where results are stored on-the-fly. Memoization involves recursive function calls that cache subproblem results upon their first encounter, ensuring each subproblem is solved only once and reused when needed.

Conversely, tabulation employs a bottom-up approach, utilizing an iterative method to systematically fill a table (or array) with subproblem solutions. This typically begins with the smallest subproblems, iteratively combining results to solve larger ones until the final solution is obtained.

In `countPathsMemoization`, the function checks if the result for a specific cell is already computed and stored in the `dp` matrix. If not, it recursively computes the result by combining the results of smaller subproblems and then stores it in the `dp` matrix. This process continues until all necessary subproblems are solved, a characteristic of memoization.

The `dp` matrix is utilized to store intermediate results, preventing redundant calculations. Each cell within `dp` corresponds to a subproblem, representing the number of paths to that cell from the origin. Recursive calls compute the number of paths to a given cell by summing the paths from the cell above and the cell to the left.

_The function boasts a time complexity of $O(m \times n)$_. This is because each cell in the `dp` matrix is computed only once, with each computation requiring constant time. Thus, the total operations are proportional to the number of cells in the matrix, namely $m \times n$.

Similarly, _the space complexity is $O(m \times n)$ due to the `dp` matrix_, which stores the number of paths for each cell. The matrix necessitates $m \times n$ space to accommodate these intermediate results.

In essence, the Dynamic Programming approach with memoization transforms the exponential time complexity of the naive Brute-Force solution into a linear complexity with respect to the matrix size. Consequently, this solution proves far more efficient and viable for larger matrices.

Finally, we need to run this code and compare it to the Brute-Force version.

```Shell
-----------------------------------
Calculating Paths in a 3x3 matrix
Average time for Brute-Force: 12494 ns
Average time for Memoization: 4685 ns
-----------------------------------
```

_Output 9: Comparison between Brute-Force, Memoization and Tabulation functions._{: class="legend"}

I ran it dozens of times and, most of the time, the memoized function was twice as fast as the Brute-Force function, and sometimes it was three times faster. Now we need to look at the Dynamic Programming solution using tabulation.

### Using Tabulation

Like I did before, the Code Fragment 18 shows the function I created to apply tabulation. The function `int countPathsTabulation(int m, int n)` uses Dynamic Programming with tabulation to count all possible paths in an $m \times n$ matrix.

```Cpp
// Function to count paths using Dynamic Programming with tabulation
int countPathsTabulation(int m, int n) {
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == 0 || j == 0) {
                dp[i][j] = 1;
            } else {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }

    return dp[m - 1][n - 1];
}
```

_Code Fragment 18 - Count all paths function using Tabulation._{: class="legend"}

The function `int countPathsTabulation(int m, int n)` counts all possible paths in an $m \times n$ matrix using Dynamic Programming with tabulation. The `dp` matrix is used to store the number of paths to each cell in a bottom-up manner, ensuring that each subproblem is solved iteratively. Each subproblem represents the number of distinct paths to a cell $(i, j)$ from the top-left corner $(0, 0)$.

The function initializes a `dp` matrix of size $m \times n$ with all values set to $0$. It then iterates over each cell in the matrix. If a cell is in the first row or first column, it sets the value to $1$ because there is only one way to reach these cells (either all the way to the right or all the way down). For other cells, it calculates the number of paths by summing the values from the cell above (`dp[i - 1][j]`) and the cell to the left (`dp[i][j - 1]`). This sum is then stored in `dp[i][j]`.

Using `std::array<std::array<int, MAX_SIZE>, MAX_SIZE>` for the `dp` matrix ensures efficient storage and retrieval of intermediate results. This tabulated approach systematically fills the `dp` matrix from the smallest subproblems up to the final solution, avoiding the overhead of recursive function calls and providing a clear and straightforward iterative solution.

_The time complexity of this function is $O(m \times n)$_. Each cell in the `dp` matrix is computed once, and each computation takes constant time, making the total number of operations proportional to the number of cells in the matrix. Similarly, _the space complexity is $O(m \times n)$_ due to the `dp` matrix, which requires $m \times n$ space to store the number of paths for each cell.

This Dynamic Programming approach with tabulation significantly improves performance compared to the naive Brute-Force solution. By transforming the problem into a series of iterative steps, it provides an efficient and scalable solution for counting paths in larger matrices.

Here's something interesting, dear reader. While using `std::vector` initially yielded similar execution times to the memoization function, switching to `std::array` resulted in a dramatic improvement. Not only did it significantly reduce memory usage, but it also made the function up to 4 times faster!

After running additional tests, the tabulation function averaged $3$ times faster execution than the original. However, there's a caveat: using `std::array` necessitates knowing the input size beforehand, which might not always be practical in real-world scenarios. Finally we can take a look in this function execution time:

```Shell
-----------------------------------
Calculating Paths in a 3x3 matrix
Average time for Brute-Force: 10453 ns
Average time for Memoization: 5348 ns
Average time for Tabulation: 1838 ns
-----------------------------------
```

The key takeaway is this: both memoization and tabulation solutions share the same time complexity. Therefore, in an interview setting, the choice between them boils down to personal preference. But if performance is paramount, tabulation (especially with `std::array` if the input size is known) is the way to go. Of course, now it's up to the diligent reader to test all the functions we've developed to solve problem "Counting All Possible Paths in a Matrix" with `std::array`. Performance has its quirks, and since there are many factors outside the programmer's control involved in execution, we always need to test in the an environment just like the production environment.

## Problem 2 Statement: Subset Sum

Given $N$ integers and $T$, determine whether there exists a subset of the given set whose elements sum up to $T$.

### Input

- An integer $N$ representing the number of integers.
- An integer $T$ representing the target sum.
- A list of $N$ integers.

### Output

- A boolean value indicating whether such a subset exists.

### Example

Input:
5 10
2 3 7 8 10

Output:
true

### Constraints

- $1 \leq N \leq 100$
- $1 \leq T \leq 1000$
- Each integer in the list is positive and does not exceed $100$.

### Analysis

The "Subset Sum" problem has already been tackled in the chapter: "Your First Dynamic Programming Problem." Therefore, our diligent reader should review the conditions presented here and see if the solution we presented for the "Two-Sum" problem applies in this case. If not, it'll be up to the reader to adapt the previous code accordingly. I'll kindly wait before we go on.

## Problem 3 Statement: Longest Increasing Subsequence

You are given an array containing $N$ integers. Your task is to determine the Longest Increasing Subsequence (LIS) in the array, where every element is larger than the previous one.

### Input

- An integer $N$ representing the number of integers.
- A list of $N$ integers.

### Output

- An integer representing the length of the Longest Increasing Subsequence.

### Example

Input:
6
5 2 8 6 3 6 9 7

Output:
4

### Constraints

- $1 \leq N \leq 1000$
- Each integer in the list can be positive or negative.

### Analysis

The "Longest Increasing Subsequence" (LIS) problem is a classic problem in Dynamic Programming, often appearing in interviews and programming competitive programmings. The goal is to find the length of the longest subsequence in a given array such that all elements of the subsequence are in strictly increasing order. _There are three main approaches to solving this problem: Brute-Force, memoization, and tabulation. Coincidentally, these are the three solutions we are studying_. So, let's go.

### Brute-Force

In the Brute-Force approach, we systematically generate all possible subsequences of the array and examine each one to determine if it's strictly increasing. The length of the longest increasing subsequence is then tracked and ultimately returned. Here's the algorithm for solving the LIS problem using Brute-Force:

1. Iteratively generate all possible subsequences of the array. (We'll reserve recursion for the memoization approach.)
2. For each subsequence, verify if it is strictly increasing.
3. Keep track of the maximum length encountered among the increasing subsequences.

Code Fragment 19 presents the function I developed using a brute-force approach.

```Cpp
// Iterative Brute-Force LIS function
int longestIncreasingSubsequenceBruteForce(const std::vector<int>& arr) {
    int n = arr.size();
    int maxLen = 0;

    // Generate all possible subsequences using bitmasking
    for (int mask = 1; mask < (1 << n); ++mask) {
        std::vector<int> subsequence;
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                subsequence.push_back(arr[i]);
            }
        }

        // Check if the subsequence is strictly increasing
        bool isIncreasing = true;
        for (int i = 1; i < subsequence.size(); ++i) {
            if (subsequence[i] <= subsequence[i - 1]) {
                isIncreasing = false;
                break;
            }
        }

        if (isIncreasing) {
            maxLen = std::max(maxLen, (int)subsequence.size());
        }
    }

    return maxLen;
}
```

_Code Fragment 19 - Interactive function to solve the LIS problem._{: class="legend"}

In the function `int longestIncreasingSubsequenceBruteForce(const std::vector<int>& arr)`, bitmasking is used to generate all possible subsequences of the array. Bitmasking involves using a binary number, where each bit represents whether a particular element in the array is included in the subsequence. For an array of size $n$, there are $2^n$ possible subsequences, corresponding to all binary numbers from $1$ to $(1 << n) - 1$. In each iteration, the bitmask is checked, and if the i-th bit is set (i.e., `mask & (1 << i)` is true), the i-th element of the array is included in the current subsequence. This process ensures that every possible combination of elements is considered, allowing the function to generate all potential subsequences for further evaluation.

For every generated subsequence, the function meticulously examines its elements to ensure they are in strictly increasing order. This involves comparing each element with its predecessor, discarding any subsequence where an element is not greater than the one before it.

Throughout the process, the function keeps track of the maximum length among the valid increasing subsequences encountered. If a subsequence's length surpasses the current maximum, the function updates this value accordingly.

While ingenious, this brute-force method has a notable drawback: _an exponential time complexity of $O(2^n _ n)$*. This arises from the $2^n$ possible subsequences and the $n$ operations needed to verify the increasing order of each. Consequently, this approach becomes impractical for large arrays due to its high computational cost.

Notice that, once again, I started by using `std::vector` since we usually don't know the size of the input dataset beforehand. Now, all that remains is to run this code and observe its execution time. Of course, my astute reader should remember that using `std::array` would likely require knowing the maximum input size in advance, but it would likely yield a faster runtime.

```Cpp
-----------------------------------
Calculating LIS in the array
Average time for LIS: 683023 ns
-----------------------------------
```

_Output 10: Execution time for LIS solution using Brute-Force._{: class="legend"}

At this point, our dear reader should have a mantra in mind: 'Don't use Brute-Force... Don't use Brute-Force.' With that said, let's delve into Dynamic Programming algorithms, keeping Brute-Force as a reference point for comparison.

### Memoization

Memoization is a handy optimization technique that remembers the results of expensive function calls. If the same inputs pop up again, we can simply reuse the stored results, saving precious time. Let's see how this applies to our LIS problem.

We can use memoization to avoid redundant calculations by storing the length of the LIS ending at each index. Here's the game plan:

1. Define a recursive function `LIS(int i, const std::vector<int>& arr, std::vector<int>& dp)` that returns the length of the LIS ending at index `i`.
2. Create a memoization array called `dp`, where `dp[i]` will store the length of the LIS ending at index `i`.
3. For each element in the array, compute `LIS(i)` by checking all previous elements `j` where `arr[j]` is less than `arr[i]`. Update `dp[i]` accordingly.
4. Finally, the maximum value in the `dp` array will be the length of the longest increasing subsequence.

Here's the implementation of the function: (Code Snippet 21)

```Cpp
// Recursive function to find the length of LIS ending at index i with memoization
int LIS(int i, const std::vector<int>& arr, std::vector<int>& dp) {
    if (dp[i] != -1) return dp[i];

    int maxLength = 1; // Minimum LIS ending at index i is 1
    for (int j = 0; j < i; ++j) {
        if (arr[j] < arr[i]) {
            maxLength = std::max(maxLength, LIS(j, arr, dp) + 1);
        }
    }
    dp[i] = maxLength;
    return dp[i];
}

// Function to find the length of the Longest Increasing Subsequence using memoization
int longestIncreasingSubsequenceMemoization(const std::vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return 0;

    std::vector<int> dp(n, -1);
    int maxLength = 1;
    for (int i = 0; i < n; ++i) {
        maxLength = std::max(maxLength, LIS(i, arr, dp));
    }

    return maxLength;
}
```

_Code Fragment 20 - Function to solve the LIS problem using recursion and Memoization._{: class="legend"}

Let's break down how these two functions work together to solve the Longest Increasing Subsequence (LIS) problem using memoization.

The `LIS(int i, const std::vector<int>& arr, std::vector<int>& dp)` recursive function calculates the length of the LIS that ends at index $i$ within the input array `arr`. The `dp` vector acts as a memoization table, storing results to avoid redundant calculations. If `dp[i]` is not $-1$, it means the LIS ending at index `i` has already been calculated and stored. In this case, the function directly returns the stored value. If `dp[i]` is $-1$, the LIS ending at index `i` has not been computed yet. The function iterates through all previous elements (`j` from 0 to `i-1`) and checks if `arr[j]` is less than `arr[i]`. If so, it recursively calls itself to find the LIS ending at index `j`. The maximum length among these recursive calls (plus 1 to include the current element `arr[i]`) is then stored in `dp[i]` and returned as the result.

The `longestIncreasingSubsequenceMemoization(const std::vector<int>& arr)` function serves as a wrapper for the `LIS` function and calculates the overall LIS of the entire array. It initializes the `dp` array with -1 for all indices, indicating that no LIS values have been computed yet. It iterates through the array and calls the `LIS` function for each index `i`. It keeps track of the maximum length encountered among the results returned by `LIS(i)` for all indices. Finally, it returns this maximum length as the length of the overall LIS of the array.

Comparing the complexity of the memoization solution with the Brute-Force solution highlights significant differences in efficiency. The Brute-Force solution generates all possible subsequences using bitmasking, which results in a time complexity of $O(2^n \cdot n)$ due to the exponential number of subsequences and the linear time required to check each one. _In contrast, the memoization solution improves upon this by storing the results of previously computed LIS lengths, reducing redundant calculations. This reduces the time complexity to $O(n^2)$, as each element is compared with all previous elements, and each comparison is done once_. The space complexity also improves from potentially needing to store numerous subsequences in the Brute-Force approach to a linear $O(n)$ space for the memoization array in the Dynamic Programming solution. Thus, the memoization approach provides a more scalable and practical solution for larger arrays compared to the Brute-Force method. What can be seen in output 22:

```Shell
-----------------------------------
Calculating LIS in the array
Average time for LIS (Brute-Force): 690399 ns
Average time for LIS (Memoization): 3018 ns
-----------------------------------
```

_Output 11: Execution time for LIS solution using Memoization._{: class="legend"}

The provided output clearly demonstrates the significant performance advantage of memoization over the Brute-Force approach for calculating the Longest Increasing Subsequence (LIS). The Brute-Force method, with its average execution time of $690,399$ nanoseconds (ns), suffers from exponential time complexity, leading to a sharp decline in performance as the input size increases.

In contrast, the memoization approach boasts an average execution time of a mere $3,018$ ns. This dramatic improvement is a direct result of eliminating redundant calculations through the storage and reuse of intermediate results. In this particular scenario, memoization is approximately $228$ times faster than Brute-Force, highlighting the immense power of Dynamic Programming techniques in optimizing algorithms that involve overlapping subproblems.

Now, let's turn our attention to the last Dynamic Programming technique we are studying: tabulation.

### Tabulation

Tabulation, a bottom-up Dynamic Programming technique, iteratively computes and stores results in a table. For the LIS problem, we create a table `dp` where `dp[i]` represents the length of the LIS ending at index `i`.

Here's a breakdown of the steps involved:

1. A table `dp` is initialized with all values set to 1, representing the minimum LIS (the element itself) ending at each index.
2. Two nested loops are used to populate the `dp` table:
   - The outer loop iterates through the array from the second element (`i` from 1 to N-1).
   - The inner loop iterates through all preceding elements (`j` from 0 to i-1).
   - For each pair (i, j), if `arr[j]` is less than `arr[i]`, it signifies that `arr[i]` can extend the LIS ending at `arr[j]`. In this case, `dp[i]` is updated to `dp[i] = max(dp[i], dp[j] + 1)`.
3. After constructing the `dp` table, the maximum value within it is determined, representing the length of the longest increasing subsequence in the array.

This brings us to Code Fragment 21, which demonstrates the implementation of this tabulation approach:

```Cpp
// Function to find the length of the Longest Increasing Subsequence using tabulation
int longestIncreasingSubsequenceTabulation(const std::vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return 0;

    std::vector<int> dp(n, 1);
    int maxLength = 1;

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (arr[i] > arr[j]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
            }
        }
        maxLength = std::max(maxLength, dp[i]);
    }

    return maxLength;
}
```

_Code Fragment 21 - Function to solve the LIS problem using recursion and Memoization._{: class="legend"}

The `int longestIncreasingSubsequenceTabulation(const std::vector<int>& arr)` function efficiently determines the length of the Longest Increasing Subsequence (LIS) in a given array using a tabulation approach.

Initially, a vector `dp` of size `n` (the array's length) is created, with all elements initialized to $1$. This signifies that the minimum LIS ending at each index is $1$ (the element itself). Additionally, a variable `maxLength` is initialized to $1$ to track the overall maximum LIS length encountered.

The function then employs nested loops to construct the `dp` table systematically. The outer loop iterates through the array starting from the second element (`i` from $1$ to `n-1`). The inner loop examines all previous elements (`j` from $0$ to `i-1`).

For each pair of elements (`arr[i]`, `arr[j]`), the function checks if `arr[i]` is greater than `arr[j]`. If so, it means `arr[i]` can extend the LIS ending at `arr[j]`. In this case, `dp[i]` is updated to the maximum of its current value and `dp[j] + 1` (representing the length of the LIS ending at `j` plus the current element `arr[i]`).

After each iteration of the outer loop, `maxLength` is updated to the maximum of its current value and `dp[i]`. This ensures that `maxLength` always reflects the length of the longest LIS found so far.

Finally, the function returns `maxLength`, which now holds the length of the overall LIS in the entire array.

In terms of complexity, the tabulation solution is on par with the memoization approach, both boasting a time complexity of $O(n^2)$. This is a substantial improvement over the Brute-Force method's exponential $O(2^n \times n)$ time complexity. The quadratic time complexity arises from the nested loops in both tabulation and memoization, where each element is compared with all preceding elements to determine potential LIS extensions.

However, _the tabulation approach has a slight edge in space complexity. While memoization requires $O(n)$ space to store the memoization table (``dp` array), tabulation only necessitates $O(n)$ space for the dp table as well._ This is because memoization might incur additional space overhead due to the recursive call stack, especially in cases with deep recursion.

The Output 12 demonstrates this performance difference. Both memoization and tabulation significantly outperform the Brute-Force approach, which takes an exorbitant amount of time compared to the other two. While the difference between memoization and tabulation is less pronounced in this specific example, tabulation can sometimes be slightly faster due to the overhead associated with recursive function calls in memoization.

```Shell
-----------------------------------
Calculating LIS in the array
Average time for LIS (Brute-Force): 694237 ns
Average time for LIS (Memoization): 2484 ns
Average time for LIS (Tabulation): 2168 ns
-----------------------------------
```

_Output 12: Execution time for LIS solution using Tabulation._{: class="legend"}

Ultimately, the choice between memoization and tabulation often comes down to personal preference and specific implementation details. Both offer substantial improvements over Brute-Force and are viable options for solving the LIS problem efficiently.

## Problem 4 Statement: Rod Cutting

Given a rod of length $n$ units and an integer array `cuts` where `cuts[i]` denotes a position you should perform a cut at, find the minimum total cost of the cuts. The cost of one cut is the length of the rod to be cut.

### Input

- An integer $n$ representing the length of the rod.
- An integer array `cuts` containing the positions of the cuts.

### Output

- An integer representing the minimum total cost of the cuts.

### Example

Input:
7
[1, 3, 4, 5]

Output:
16

### Constraints

- $1 \leq n \leq 1000$
- $1 \leq cuts[i] < n$
- The array `cuts` can have at most $n-1$ elements.

### Analysis

## Problem 5 Statement: Longest Common Subsequence

You are given strings $s$ and $t$. Find the length of the longest string that is a subsequence of both $s$ and $t$.

### Input

- Two strings $s$ and $t$.

### Output

- An integer representing the length of the Longest Common Subsequence.

### Example

Input:
"abcde"
"ace"

Output:
3

### Constraints

- $1 \leq |s|, |t| \leq 1000$

## Problem 6 Statement: Longest Palindromic Subsequence

Finding the Longest Palindromic Subsequence (LPS) of a given string.

### Input

- A string $s$.

### Output

- An integer representing the length of the Longest Palindromic Subsequence.

### Example

Input:
"bbbab"

Output:
4

### Constraints

- $1 \leq |s| \leq 1000$

## Problem 7 Statement: Edit Distance

The edit distance between two strings is the minimum number of operations required to transform one string into the other. The allowed operations are ["Add", "Remove", "Replace"].

### Input

- Two strings $s$ and $t$.

### Output

- An integer representing the minimum number of operations required to transform one string into the other.

### Example

Input:
"sunday"
"saturday"

Output:
3

### Constraints

- $1 \leq |s|, |t| \leq 1000$

## Problem 8 Statement: Coin Change Problem

Given an array of coin denominations and a target amount, find the minimum number of coins needed to make up that amount.

### Input

- An integer $T$ representing the target amount.
- A list of integers representing the coin denominations.

### Output

- An integer representing the minimum number of coins needed to make up the target amount.

### Example

Input:
11
[1, 2, 5]

Output:
3

### Constraints

- $1 \leq T \leq 1000$
- Each coin denomination is a positive integer not exceeding $100$.
- The list of coin denominations can have at most $100$ elements.

## Problem 9 Statement: 0-1 Knapsack

Given $W$, $N$, and $N$ items with weights $w_i$ and values $v_i$, what is the maximum $\sum_{i=1}^{k} v_i$ for each subset of items of size $k$ ($1 \leq k \leq N$) while ensuring $\sum_{i=1}^{k} w_i \leq W$?

### Input

- An integer $W$ representing the maximum weight capacity of the knapsack.
- An integer $N$ representing the number of items.
- Two lists of $N$ integers: one for weights and one for values.

### Output

- An integer representing the maximum value that can be obtained without exceeding the weight capacity.

### Example

Input:
50 3
[10, 20, 30]
[60, 100, 120]

Output:
220

### Constraints

- $1 \leq W \leq 1000$
- $1 \leq N \leq 100$
- Each weight and value is a positive integer not exceeding $1000$.

## Problem 10 Statement: Longest Path in a Directed Acyclic Graph (DAG)

Finding the longest path in a Directed Acyclic Graph (DAG).

### Input

- An integer $N$ representing the number of nodes.
- A list of tuples representing the edges of the graph, where each tuple $(u, v)$ indicates an edge from node $u$ to node $v$.

### Output

- An integer representing the length of the longest path in the DAG.

### Example

Input:
5
[(1, 2), (2, 3), (3, 4), (4, 5)]

Output:
4

### Constraints

- $1 \leq N \leq 1000$
- The number of edges does not exceed $N(N-1)/2$.

## Problem 11 Statement: Traveling Salesman Problem (TSP)

Given a list of cities and the distances between each pair of cities, find the shortest possible route that visits each city exactly once and returns to the origin city.

### Input

- An integer $N$ representing the number of cities.
- A 2D list of integers representing the distances between each pair of cities.

### Output

- An integer representing the minimum distance of the round trip.

### Example

Input:
4
[[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

Output:
80

### Constraints

- $1 \leq N \leq 10$
- Each distance is a non-negative integer not exceeding $1000$.

## Problem 12 Statement: Matrix Chain Multiplication

Given a sequence of matrices, find the most efficient way to multiply these matrices together. The problem is not actually to perform the multiplications, but merely to decide in which order to perform the multiplications.

### Input

- An integer $N$ representing the number of matrices.
- A list of $N+1$ integers representing the dimensions of the matrices.

### Output

- An integer representing the minimum number of scalar multiplications needed.

### Example

Input:
4
[10, 20, 30, 40, 30]

Output:
30000

### Constraints

- $1 \leq N \leq 100$
- Each dimension is a positive integer not exceeding $1000$.

## Problem 3 Statement: Longest Increasing Subsequence

You are given an array containing $N$ integers. Your task is to determine the Longest Increasing Subsequence (LIS) in the array, where every element is larger than the previous one.

### Input

- An integer $N$ representing the number of integers.
- A list of $N$ integers.

### Output

- An integer representing the length of the Longest Increasing Subsequence.

### Example

Input:
8
5 2 8 6 3 6 9 7

Output:
4

### Constraints

- $1 \leq N \leq 1000$
- Each integer in the list can be positive or negative.

## Notes and References

[:1] This ideia come from [Introduction to Dynamic Programming](https://cp-algorithms.com/dynamic_programming/intro-to-dp.html)

[:2] Most of this table came from [Introduction to Dynamic Programming](https://cp-algorithms.com/dynamic_programming/intro-to-dp.html)