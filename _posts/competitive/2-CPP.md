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
lastmod: 2024-10-02T16:58:57.281Z
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

# 2. C++ Here the Journey Begins

This chapter will focus on the fundamental building blocks of C++20. We start with vectors and matrices, the backbone of many algorithms. Handling arrays with care ensures faster code and more efficient memory use. Along the way, we will see techniques for handling input and output, whether through files, the keyboard, or the monitor. Each small detail in managing these operations can have a big impact on the final performance of a program.

We will also explore loops, both `for` and `while`, and how they can be controlled and optimized. These are the workhorses of any serious C++ program, and learning to use them well is key. Other features of C++20, like range-based loops and concise syntax improvements, will also be introduced. Throughout this section, we will be mindful of how each decision affects speed and clarity. You will see how simple, precise code matters when performance is on the line.

We will begin with input and output operations, focusing primarily on file reading and writing. In competitive programming, reading input efficiently is often more critical than writing output, as programs typically process large datasets. Understanding how to handle file input with speed and precision will be our priority, ensuring that we minimize bottlenecks and streamline performance. Writing output is important too, but for now, reading will take center stage.

## 2.1. Optimizing File I/O

In competitive programming contests, especially with large datasets, programs often need to read input from big files.

In C++, file input and output (I/O) operations are managed using classes from the `<fstream>` library. The main classes are `std::ifstream`, `std::ofstream`, and `std::fstream`. These classes serve different purposes: reading, writing, and both reading and writing.

- `std::ifstream`: Used for reading from files.
- `std::ofstream`: Used for writing to files.
- `std::fstream`: Used for both reading and writing to files.

The `std::ifstream` class reads files. It is only for input. It inherits from `std::istream`, the main class for input in C++. Use `std::ifstream` to open a file and read its data. You can read line by line or in parts. It is straightforward and efficient. This class also checks file status and handles errors. It is a basic tool for reading in C++.

In your code, use `std::ifstream` to open a text file and read its contents:

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

### 2.1.1. Reading Lines More Efficiently\*\*

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

## 2.2. Competitive Programming and File I/O

There are faster ways to open and handle files in C++, especially when dealing with large data sets in competitive programming. These techniques can speed up file processing.

Use `std::ios::sync_with_stdio(false);` to disable the synchronization between C++ streams and C streams (`stdio` functions). This makes input and output faster because it removes the overhead of syncing with C-style input/output.

Turn off the synchronization with `cin.tie(nullptr);`. This disconnects `cin` from `cout`, so `cout` doesn’t flush every time `cin` is used. This can save time when reading and writing a lot of data.

Use larger buffers when reading and writing to minimize the number of operations. Reading a chunk of data at once, rather than line by line, can make your program faster.

Combine `std::ios::in | std::ios::out | std::ios::binary` when opening files to read and write in binary mode, reducing the time spent on formatting operations. These tweaks make your file operations lean and quick, perfect for big data tasks.

### 2.2.1. Use Manual Buffering

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

### 2.2.2 Using `mmap` for Faster File I/O in Unix-Based Systems

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

### 2.2.3 Using Asynchronous I/O with `std::future` and `std::async` in C++20

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

### 2.2.4. Summary of Efficient Techniques for File I/O

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

## 2.3. Fast Command-Line I/O

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

## 2.4. Boosting I/O Efficiency for Bulk Data

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

## 2.5. The Sad Trick: Namespaces

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

### 2.5.1 `using namespace std;`

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

In competitive programming, mastering the basics of C++20 is essential for writing efficient and optimized code. This section focuses on fundamental control structures like loops and essential data structures such as vectors and matrices. Both elements play a important role in solving problems under time constraints.

C++20 introduces several features that enhance the flexibility and performance of loops. Techniques like range views and parallel execution allow programmers to process data with greater efficiency. Whether you are dealing with small arrays or large datasets, choosing the right loop can significantly impact the runtime of your solution.

Alongside loops, vectors and matrices serve as the foundation for storing and manipulating data. Understanding how to effectively use these data structures, combined with modern C++ features, allows you to handle complex computations with ease.

In the following sections, we will explore these elements in-depth, providing examples and performance considerations to help you develop competitive programming skills using C++20.

## 2.6. Working with Vector and Matrix

Vectors are flexible. They change size and are easy to use. You can insert, remove, and resize them with little effort. They work well for many tasks in competitive programming. Vectors hold one-dimensional data, but can also represent matrices. These two-dimensional vectors are often used for grids, tables, or simulations.

Matrices, built from vectors, handle multi-dimensional problems. They are good for game boards, adjacency matrices, and dynamic programming. You can change rows and columns, transpose the matrix, or access submatrices. Vectors and matrices give you control over how you store and process data. Starting with vectors.

In C++20, vectors have several important features that make them a powerful tool for managing collections of elements. First, vectors dynamically resize. This means they grow automatically when you add elements beyond their current capacity. You don’t need to manually manage memory like you would with arrays.

Vectors also provide random access. You can access any element by its index using `[]`, just as you would with a regular array. This makes it easy to work with elements directly without needing to traverse the vector.

### 2.6.1. Vetores, basic operations

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

### 2.6.2 Sorting the Vector

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

### 2.63 Vectors as Inputs and Outputs

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

#### 2.6.3.1 Optimized Version Using `fread` and `putchar` with Command-Line File Input

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

## 2.7 Using Matrices

In C++20, matrices are typically represented as vectors of vectors (`std::vector<std::vector<T>>`), where each inner vector represents a row of the matrix. This approach allows for dynamic sizing and easy manipulation of multi-dimensional data, making matrices ideal for problems involving grids, tables, or any 2D structure.

Matrices in C++ offer flexibility in managing data: you can resize rows and columns independently, access elements using intuitive indexing, and leverage standard vector operations for rows. Additionally, the use of `ranges` and `views` introduced in C++20 boosts the ability to iterate and transform matrix data more expressively and efficiently.

_The use of matrices is common in competitive programming for tasks such as implementing dynamic programming tables, graph adjacency matrices, or performing transformations on 2D data. With the powerful capabilities of C++20's STL, matrices become a highly adaptable and efficient way to handle complex, multi-dimensional computations in a structured manner_.

### 2.7.1 Creating and Filling a Matrix

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

### 2.7.2 Displaying the Matrix

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

### 2.2.8 Inserting Elements at a Specific Position

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

### 2.7.4 Removing the Last Element and a Specific Element

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

### 2.7.5 Creating a New Vector with a Default Value

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

### 2.7.6 Resizing and Filling with Random Values

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

This code resizes the matrix to 3x3 and fills it with random values between 1 and 2.10.

### 2.7.7 Sorting Matrices by Rows and Columns

In C++20, we can sort matrices (represented as vectors of vectors) both by rows and by columns. Here are examples of how to do both:

#### 2.7.7.1 Sorting by Rows

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

#### 2.7.7.2 Sorting by Columns

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

### 2.7.8 Optimizing Matrix Input and Output in Competitive Programming

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

#### 2.7.8.1 Optimized Reading with `fread`

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

#### 2.7.8.2 Optimized Output with `putchar_unlocked`

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

#### 2.7.8.3 Complexity Analysis

The time complexity for reading and writing matrices is $O(nm)$, where $n$ and $m$ are the dimensions of the matrix. The space complexity is also $O(nm)$, as we store the entire matrix in memory. However, the constant factors are significantly reduced compared to standard I/O methods, leading to faster execution times in practice.

#### 2.7.8.4 Using `mmap` on Unix Systems for Matrices

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

## 2.8 Using Span and Ranges

In the fast-paced world of competitive programming and high-performance computing, efficient data manipulation is paramount. C++20 introduces two powerful features - `std::span` and `std::ranges` for that.

These features are particularly important because they address common performance bottlenecks in data-intensive applications. `std::span` provides a lightweight, non-owning view into contiguous data, reducing unnecessary copying and allowing for flexible, efficient data access. `std::ranges`, on the other hand, offers a unified, composable interface for working with sequences of data, enabling more intuitive and often more performant algorithm implementations. Together, they form a potent toolkit for developers seeking to push the boundaries of what's possible in terms of code efficiency and elegance in C++.

### 2.8.1 Using `std::span`

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

#### 2.8.1.1 Efficient Use Cases for `std::span`

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

#### 2.8.1.2 Comparing `std::span` to Traditional Methods

| Feature          | `std::vector`           | Raw Pointers          | `std::span`     |
| ---------------- | ----------------------- | --------------------- | --------------- |
| Memory Ownership | Yes                     | No                    | No              |
| Memory Overhead  | High (allocates memory) | Low                   | Low             |
| Bounds Safety    | High                    | Low                   | High            |
| Compatibility    | Works with STL          | Works with raw arrays | Works with both |

Unlike `std::vector`, which manages its own memory, `std::span` does not allocate or own memory. This is similar to raw pointers but with added safety since `std::span` knows its size. `std::span` is safer than raw pointers because it carries bounds information, helping avoid out-of-bounds errors. While raw pointers offer flexibility, they lack the safety features provided by modern C++.

#### 2.8.1.3 Practical Application: Using `std::span` in Competitive Programming

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

## 2.9 Efficient Data Manipulation with `std::ranges` in C++20

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

2.10. Loops the Heart of All Competitive Programming

Loops are, without a doubt, the most important part of any code, whether for competitive programming, high-performance applications, or even solving academic problems. Most programming languages offer more than one way to implement loops. In this text, since Python is only our pseudocode language, we will focus on studying loops in C++.

## 2.10. Deep Dive into `for` Loops in Competitive Programming

C++ provides several ways to iterate over elements in a vector, using different types of `for` loops. In this section, we will explore the various `for` loop options available in C++20, discussing their performance and code-writing efficiency. We will also analyze which loops are best suited for competitive programming based on input size—whether dealing with small or large datasets.

### 2.10.1 `for` Loop with Iterator

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

### 2.10.2. Classic `for` Loop with Index

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

### 2.10.3. Range-Based `for-each` with Constant Reference

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

### 2.10.4. Range-Based `for-each` by Value

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

### 2.10.5. `for` Loop with Range Views (C++20)

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

### 2.10.6. Parallel `for` Loop (C++17/C++20)

While not a traditional `for` loop, using parallelism in loops is a powerful feature introduced in C++17 and further improved in C++20.

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

### 2.10.7. Optimal `for` Loops for Competitive Programming

Choosing the right type of `for` loop in competitive programming depends largely on input size and the specific use case. The following table summarizes the best choices for different scenarios:

| Input Size      | Best `for` Loop Option                                             | Reasoning                                                                                            |
| --------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Small           | Range-Based `for-each` with Constant Reference                     | Offers minimal syntax, high readability, and avoids copies, making it fast and efficient.            |
| Small           | Classic `for` Loop with Index                                      | Provides precise control over the index, useful when index manipulation or modification is required. |
| Large           | Iterator-Based `for` Loop                                          | Highly efficient for large datasets due to minimal memory overhead and optimized performance.        |
| Large           | Parallel `for` Loop with `std::for_each` and `std::execution::par` | Ideal for computationally heavy tasks on large datasets, leveraging multiple threads to parallelize. |
| Transformations | `for` Loop with Range Views (C++20)                                | Ideal for processing subsets or transformations of data without creating extra copies.               |

## 2.11 Now the `while` Loop which we all love

The `while` loop is another fundamental control structure in C++ that is often used in competitive programming. It repeatedly executes a block of code as long as a specified condition evaluates to true. In this section, we will explore the different use cases for `while` loops, their performance considerations, and scenarios where they may be preferable to `for` loops. We will also examine their application with both small and large datasets.

### 2.11.1. Basic `while` Loop

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

### 2.11.2. `while` Loop with Complex Conditions

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

### 2.11.3. Infinite `while` Loops

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

### 2.11.4. `do-while` Loop

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

### 2.11.5. `while` Loop with Early Exit

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

### 2.11.6. Combining `while` with Multiple Conditions

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

### 2.11.7. Optimal `while` Loops for Competitive Programming

Choosing the right type of `while` loop depends on the nature of the input and the complexity of the condition. The following table summarizes the optimal choices for different input sizes:

| Input Size | Best `while` Loop Option                   | Reasoning                                                                                                                  |
| ---------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Small      | Basic `while` Loop                         | Offers straightforward control over iteration with minimal overhead and is easy to implement.                              |
| Small      | `do-while` Loop                            | Ensures at least one execution of the loop, which is crucial for cases where the first iteration is essential.             |
| Large      | `while` with Early Exit                    | Improves performance by terminating the loop early when a specific condition is met, saving unnecessary iterations.        |
| Large      | `while` with Complex Conditions            | Allows dynamic and flexible exit conditions, making it suitable for large datasets with evolving parameters.               |
| Continuous | Infinite `while` Loop with Explicit Breaks | Best for situations where the exact number of iterations is unknown and depends on external factors or dynamic conditions. |

## 2.12 Special Loops in C++20 for Competitive Programming

In C++20, several advanced looping techniques have been introduced, each offering unique ways to improve code efficiency and readability. While some of these techniques provide remarkable performance optimizations, not all are well-suited for competitive programming. competitive programmings often involve handling dynamic inputs and generating outputs within strict time limits, so techniques relying heavily on compile-time computation are less practical. This section focuses on the most useful loop structures for competitive programmings, emphasizing runtime efficiency and adaptability to varying input sizes.

### 2.12.1. Range-Based Loops with `std::ranges::views`

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

### 2.12.2. Parallel Loops with `std::for_each` and `std::execution::par`

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

## 2.13. `constexpr` Loops

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

### 2.13.1 Considerations for competitive programmings

While constexpr loops are not suitable for processing dynamic inputs directly, they can be strategically used to create lookup tables or pre-compute values that are then utilized during runtime calculations. This can be particularly useful in problems involving mathematical sequences, combinatorics, or other scenarios where certain calculations can be predetermined. _However, it's important to balance the use of pre-computed data with memory constraints, as large lookup tables might exceed memory limits in some competitive programming environments_.

## 2.14. Early Exit Loops

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

Early exit loops improve efficiency by terminating as soon as a specified condition is met, thus avoiding unnecessary iterations. This approach helps save time, especially when the loop would otherwise continue without contributing to the result. This technique is particularly useful in search problems. By exiting the loop early when a target value is found, it can improve performance, reducing the overall execution time.

_Early exit loops are highly practical, as they allow a solution to be reached without the need to examine all the data. By cutting down unnecessary iterations, they help reduce execution time, making them particularly useful in scenarios where a result can be determined quickly based on partial input._

## 2.15. Indexed Loops with Range-Based `for`

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

Indexed loops are well-suited for dynamic access, offering the flexibility required for more complex iteration logic. This makes them ideal for scenarios where direct control over the loop's behavior is necessary.

However, they are less expressive compared to range-based loops. While they provide detailed control, they tend to be more verbose and less concise than the streamlined syntax offered by range-based alternatives.

## 2.16. Standard Library Algorithms (`std::for_each`, `std::transform`)

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

Additionally, these functions are concise and clear, providing a clean and expressive syntax to apply operations on containers. This simplicity improve code readability while maintaining high performance, making them ideal for competitive programming.

Standard library algorithms are great for transformation tasks, allowing you to apply operations on container elements with minimal code. They maximize efficiency while keeping the implementation simple and concise, making them particularly effective for handling transformations in competitive programming scenarios.

## 2.17 Summary Table of Useful Loop Techniques for competitive programmings

| Technique                                 | Best Use Case                            | Efficiency Considerations                                                          |
| ----------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------- |
| `std::ranges::views`                      | Transforming or filtering large datasets | Lazily evaluated operations reduce memory overhead and improve runtime efficiency. |
| Parallel Loops with `std::execution::par` | Large computational tasks                | Parallelism significantly improves performance for large, independent tasks.       |
| Early Exit Loops                          | Search or conditional exit problems      | Avoids unnecessary iterations, improving efficiency in scenarios with early exits. |
| Indexed Loops                             | Precise control over iteration           | Offers flexibility and control for complex iteration logic or index manipulation.  |
| Standard Library Algorithms               | Applying transformations or actions      | Well-optimized algorithms that simplify code and improve performance.              |

**Techniques Not Recommended for competitive programmings**:

| Technique         | Reasoning                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `constexpr` Loops | Compile-time only, cannot handle dynamic input, thus impractical for runtime competitive programming problems. |
