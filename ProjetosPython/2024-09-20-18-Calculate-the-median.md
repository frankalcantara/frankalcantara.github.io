---
author: Frank
beforetoc: '[Anterior](2024-09-20-17-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-19-Sem-T%C3%ADtulo.md)'
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
title: Calculate the median
toc: true
---
# Calculate the median
sorted_heights = sorted(final_heights[:-1])
if N % 2 == 1:
    median = sorted_heights[N // 2]
else:
    median = (sorted_heights[(N - 1) // 2] + sorted_heights[N // 2]) // 2

print("Median height:", median)*Output**: [2, 5, 3, 0]
```

**Algorithm Implementation**: C++20

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

// Function to perform range update on the difference array
void range_update(vector<int>& diff, int l, int r, int x) {
    diff[l] += x;
    if (r + 1 < static_cast<int>(diff.size())) {
        diff[r + 1] -= x;
    }
}

// Function to reconstruct the final heights from the difference array
vector<int> reconstruct_heights(const vector<int>& diff) {
    vector<int> heights(diff.size());
    partial_sum(diff.begin(), diff.end(), heights.begin());
    return heights;
}

int main() {
    // Hardcoded inputs based on the example in the image
    int N = 7;  // Number of stacks
    int K = 4;  // Number of instructions
    vector<pair<int, int>> instructions = {
        {3, 5},
        {5, 5},
        {2, 4},
        {4, 6}
    };

    vector<int> diff(N + 1, 0);  // Difference array initialized with 0s

    // Apply all instructions
    for (const auto& [A, B] : instructions) {
        range_update(diff, A - 1, B - 1, 1);  // -1 for 0-based indexing
    }

    vector<int> final_heights = reconstruct_heights(diff);

    // Print final heights for verification
    cout << "Final heights: ";
    for (int height : final_heights) {
        cout << height << " ";
    }
    cout << endl;

    // Sort the heights to find the median
    sort(final_heights.begin(), final_heights.end());

    // Calculate the median
    int median;
    if (N % 2 == 1) {
        median = final_heights[N / 2];
    } else {
        median = (final_heights[(N - 1) / 2] + final_heights[N / 2]) / 2;
    }

    cout << "Median height: " << median << endl;

    return 0;
}
```

**Advantages and Limitations**:

The Difference Array algorithm is highly efficient for handling multiple range updates. It allows constant time updates, $O(1)$, which makes it particularly useful in scenarios with a large number of updates. This efficiency makes the algorithm well-suited for large-scale problems that require numerous updates.

However, the algorithm is not ideal for frequent individual element queries, as reconstructing the array after updates takes $O(n)$. Additionally, to access individual elements after performing multiple updates, it requires a full array reconstruction, which can be a drawback in cases where immediate access to array elements is needed.

#### Problem Example: "Humidity Levels in a Greenhouse" (Problem 1)

You are responsible for monitoring and adjusting the humidity levels in a greenhouse that contains various plants. The greenhouse has a set of humidity sensors, represented by an array $humidity$, where each position in the array corresponds to the reading of a sensor.

Throughout the day, you receive a series of adjustment instructions called $adjustments$. Each adjustment instruction is represented by a pair $[\text{adjustment,} \, \text{sensor}\_index]$, where $adjustment$ indicates the change that must be made to the reading of the sensor located at $s\text{sensor}\_index$.

After each adjustment, you must verify the sum of the humidity levels that are within an acceptable range (i.e., are even).

Your goal is to calculate this sum for each adjustment and report it in a final report.

**Example 1:**

**Input**: $humidity = [45, 52, 33, 64]$, $adjustments = [[5,0],[-20,1],[-14,0],[18,3]]$
**Output**: $[166,146,132,150]$
Explanation: Initially, the array is $[45,52,33,64]$.
After adding $5$ to $humidity[0]$, the array becomes $[50,52,33,64]$, and the sum of even values is $50 + 52 + 64 = 166$.
After adding $-20$ to $humidity[1]$, the array becomes $[50,32,33,64]$, and the sum of even values is $50 + 32 + 64 = 146$.
After adding $-14$ to $humidity[0]$, the array becomes $[36,32,33,64]$, and the sum of even values is $36 + 32 + 64 = 132$.
After adding $18$ to $humidity[3]$, the array becomes $[36,32,33,82]$, and the sum of even values is $36 + 32 + 82 = 150$.

**Example 2**:

**Input**: $humidity = [40]$, $adjustments = [[12,0]]$
**Output**: $[52]$

**Example 3**:

**Input**: $humidity = [30, 41, 55, 68, 72]$, $adjustments = [[10,0],[-15,2],[22,1],[-8,4],[5,3]]$
**Output**: $[180,220,220,212,144]$

**Explanation**:

- Initially, the array is $[30,41,55,68,72]$.
- After adding $10$ to $humidity[0]$, the array becomes $[40,41,55,68,72]$, and the sum of the even values is $40 + 68 + 72 = 180$.
- After adding $-15$ to $humidity[2]$, the array becomes $[40,41,40,68,72]$, and the sum of the even values is $40 + 40 + 68 + 72 = 220$.
- After adding $22$ to $humidity[1]$, the array becomes $[40,63,40,68,72]$, and the sum of the even values is $40 + 40 + 68 + 72 = 220$.
- After adding $-8$ to $humidity[4]$, the array becomes $[40,63,40,68,64]$, and the sum of the even values is $40 + 40 + 68 + 64 = 212$.
- After adding $5$ to $humidity[3]$, the array becomes $[40,63,40,73,64]$, and the sum of the even values is $40 + 40 + 64 = 144$.

**Constraints:**

- The number of sensors in the greenhouse is at least $1$ and at most $10,000$.
- Each humidity reading is between $-10,000$ and $10,000$.
- The number of adjustments during the day can vary between $1$ and $10,000$.
- Each adjustment can increase or decrease the sensor reading by up to $10,000$ units.

##### Naïve Solution

**Algorithm**:

1. Initialize an empty list called `results` to store the sums of even values after each adjustment.

2. For each adjustment $[adjustment, sensorIndex]$ in the `adjustments` list:

   a. Update the corresponding sensor's value in the `humidity` array:
   $humidity[sensorIndex] = humidity[sensorIndex] + adjustment$

   b. Calculate the sum of even values in the `humidity` array:

   - Initialize a variable `even_sum` to 0
   - For each value $h$ in `humidity`:
     - If $h$ is even (i.e., $h \bmod 2 = 0$), add $h$ to `even_sum`

   c. Append `even_sum` to the `results` list

3. Return the `results` list

**Implementation**: Pseudo code.

```python
def calculate_even_sum_after_adjustments(humidity, adjustments):
    # Inicializa uma lista para armazenar os resultados
    results = []

    # Para cada ajuste na lista de ajustes
    for adjustment, sensor_index in adjustments:
        # Atualiza o valor do sensor correspondente no array de umidade
        humidity[sensor_index] += adjustment

        # Calcula a soma dos valores pares na lista de umidade
        even_sum = 0
        for h in humidity:
            if h % 2 == 0:  # Verifica se o valor é par
                even_sum += h

        # Adiciona a soma atual dos valores pares na lista de resultados
        results.append(even_sum)

    # Retorna a lista de resultados
    return results

