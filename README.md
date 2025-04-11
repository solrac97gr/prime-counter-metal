# Metal Prime Number Calculator

This project uses Apple's Metal API to efficiently count prime numbers within a specified range using the Sieve of Eratosthenes algorithm implemented on the GPU.

## Overview

The application leverages the parallel processing capabilities of the GPU to perform a high-performance prime number calculation. It implements a bit-packed sieve approach that optimizes memory usage by representing 32 numbers in each 32-bit unsigned integer.

## Features

- **GPU Accelerated**: Uses Metal compute shaders to parallelize the prime number sieve algorithm
- **Memory Efficient**: Implements a bit-packed sieve (32 numbers per integer)
- **Accurate Results**: Carefully handles edge cases and atomic operations for precise counting
- **Performance Optimized**: Uses multiple kernel passes for initialization, marking, and counting

## Implementation Details

The algorithm is implemented in three main steps:

1. **Initialize Sieve**: All numbers are initially marked as prime (1), except for 0 and 1
2. **Mark Multiples**: Each thread representing a prime number marks all of its multiples as non-prime
3. **Count Primes**: The remaining marked numbers are counted in parallel

## Requirements

- macOS with Metal support
- Swift 5.0+
- Xcode 12.0+ (for building)

## Usage

To run the program:

```bash
swift primes.swift
```

You can modify the `range` value in the code to count primes up to a different limit.

## Performance

The implementation is optimized for counting large ranges of prime numbers. For the default range of 100 million, it should complete in just a few seconds on modern hardware, significantly faster than equivalent CPU implementations.

## How It Works

### Metal Shader Functions

The Metal code includes three main kernel functions:

- `initSieve`: Initializes the bit-packed sieve with all numbers marked as potential primes
- `markMultiples`: Each thread representing a prime number ≤ √range marks all of its multiples as composite
- `countPrimes`: Counts the remaining prime numbers in parallel and accumulates the total

### Memory Model

The implementation uses a bit-packed sieve where each bit represents a number. This approach:

- Reduces memory usage by 32x compared to using one integer per number
- Uses atomic operations to safely handle concurrent modifications
- Improves memory access patterns for better GPU performance

## Customization

You can customize the `range` variable to compute prime counts for different upper bounds.
