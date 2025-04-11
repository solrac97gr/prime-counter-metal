import Foundation
import Metal

// Function to count primes using Metal
func countPrimesWithMetal(range: UInt32) {
    print("Counting primes from 0 to \(range) using Metal...")
    let startTime = CFAbsoluteTimeGetCurrent()

    // Create the Metal device
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("Metal is not supported on this device")
    }

    // Create a command queue
    guard let commandQueue = device.makeCommandQueue() else {
        fatalError("Could not create command queue")
    }

    // Load the Metal shader library
    let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        // For boolean array in a bit-packed format
        // Each integer stores 32 boolean values
        kernel void primeSieve(
            device uint* sieve [[buffer(0)]],      // Bit array for the sieve
            constant uint& range [[buffer(1)]],    // The maximum number to check
            uint id [[thread_position_in_grid]],
            uint threads [[threads_per_grid]])
        {
            // Each thread handles a subset of potential prime numbers
            // Start from 2 (first prime) and skip 0 and 1

            // Calculate the chunk size for each thread
            uint sqrtRange = sqrt(float(range));

            // First pass: find primes up to sqrt(range)
            if (id == 0) {
                // Initialize all to prime (1)
                for (uint i = 0; i < (range / 32) + 1; i++) {
                    sieve[i] = 0xFFFFFFFF; // All bits set to 1 (prime)
                }

                // 0 and 1 are not prime
                sieve[0] &= ~(1 << 0);  // Clear bit 0
                sieve[0] &= ~(1 << 1);  // Clear bit 1

                // Sieve of Eratosthenes algorithm (sequential for sqrt part)
                for (uint i = 2; i <= sqrtRange; i++) {
                    // If i is prime
                    if (sieve[i / 32] & (1 << (i % 32))) {
                        // Mark all multiples of i as non-prime
                        for (uint j = i * i; j <= range; j += i) {
                            sieve[j / 32] &= ~(1 << (j % 32));
                        }
                    }
                }
            }

            // Synchronize threads to ensure first pass is complete
            threadgroup_barrier(mem_flags::mem_device);

            // Second pass: Parallelize the sieving for numbers > sqrt(range)
            // Divide the remaining work across threads
            uint threadsNeeded = sqrtRange - 1; // Number of primes ≤ sqrt(range)
            if (id <= threadsNeeded && id >= 2) { // Skip non-prime threads 0,1
                if (sieve[id / 32] & (1 << (id % 32))) { // If this thread ID is a prime number
                    // Mark multiples as non-prime, starting from id*id
                    for (uint j = id * id; j <= range; j += id) {
                        // Atomic operation to prevent race conditions
                        atomic_fetch_and_explicit(
                            (device atomic_uint*)(sieve + (j / 32)),
                            ~(1 << (j % 32)),
                            memory_order_relaxed
                        );
                    }
                }
            }
        }

        // Kernel to count prime numbers identified by the sieve
        kernel void countPrimes(
            device const uint* sieve [[buffer(0)]],   // Bit array with the sieve results
            constant uint& range [[buffer(1)]],       // The maximum number to check
            device atomic_uint* totalCount [[buffer(2)]],  // For storing the total count
            uint id [[thread_position_in_grid]],
            uint threads [[threads_per_grid]])
        {
            // Determine chunk size per thread
            uint numbersPerThread = (range + threads - 1) / threads;
            uint start = id * numbersPerThread;
            uint end = min(start + numbersPerThread, range + 1);

            // Count primes in this thread's range
            uint localCount = 0;

            for (uint i = start; i < end; i++) {
                if (sieve[i / 32] & (1 << (i % 32))) {
                    localCount++;
                }
            }

            // Add this thread's count to the global count
            atomic_fetch_add_explicit(totalCount, localCount, memory_order_relaxed);
        }
        """

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: shaderSource, options: nil)
    } catch {
        fatalError("Failed to create Metal library: \(error)")
    }

    // Create the compute pipeline for the prime sieve
    guard let sieveFunction = library.makeFunction(name: "primeSieve") else {
        fatalError("Failed to find primeSieve kernel function")
    }

    let sievePipelineState: MTLComputePipelineState
    do {
        sievePipelineState = try device.makeComputePipelineState(function: sieveFunction)
    } catch {
        fatalError("Failed to create sieve compute pipeline state: \(error)")
    }

    // Create the compute pipeline for counting primes
    guard let countFunction = library.makeFunction(name: "countPrimes") else {
        fatalError("Failed to find countPrimes kernel function")
    }

    let countPipelineState: MTLComputePipelineState
    do {
        countPipelineState = try device.makeComputePipelineState(function: countFunction)
    } catch {
        fatalError("Failed to create count compute pipeline state: \(error)")
    }

    // Calculate buffer size for sieve (bit-packed, so 32 numbers per uint)
    let sieveSize = (Int(range) / 32) + 1
    let sieveBufferSize = sieveSize * MemoryLayout<UInt32>.size

    // Create buffers
    let sieveBuffer = device.makeBuffer(length: sieveBufferSize, options: .storageModeShared)!
    let rangeBuffer = device.makeBuffer(
        bytes: [range], length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    let countBuffer = device.makeBuffer(
        length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

    // Initialize count to 0
    let countPtr = countBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
    countPtr.pointee = 0

    // Create a command buffer
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        fatalError("Could not create command buffer")
    }

    // Create a compute command encoder for the sieve
    guard let sieveEncoder = commandBuffer.makeComputeCommandEncoder() else {
        fatalError("Could not create compute command encoder")
    }

    // Configure the sieve encoder
    sieveEncoder.setComputePipelineState(sievePipelineState)
    sieveEncoder.setBuffer(sieveBuffer, offset: 0, index: 0)
    sieveEncoder.setBuffer(rangeBuffer, offset: 0, index: 1)

    // Calculate grid and threadgroup sizes
    let sqrtRange = UInt32(sqrt(Float(range)))
    let threadExecutionWidth = sievePipelineState.threadExecutionWidth
    let threadsPerGrid = Int(max(sqrtRange + 1, 64))  // Convert to Int
    let threadsPerThreadgroup = min(threadExecutionWidth, threadsPerGrid)  // Now both are Int

    sieveEncoder.dispatchThreads(
        MTLSize(width: threadsPerGrid, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
    )

    sieveEncoder.endEncoding()

    // Create a compute command encoder for counting
    guard let countEncoder = commandBuffer.makeComputeCommandEncoder() else {
        fatalError("Could not create compute command encoder for counting")
    }

    // Configure the count encoder
    countEncoder.setComputePipelineState(countPipelineState)
    countEncoder.setBuffer(sieveBuffer, offset: 0, index: 0)
    countEncoder.setBuffer(rangeBuffer, offset: 0, index: 1)
    countEncoder.setBuffer(countBuffer, offset: 0, index: 2)

    // The counting can be parallelized more effectively
    let countThreadsPerGrid = min(1024, Int(range) + 1)  // Convert to Int
    let countThreadsPerThreadgroup = min(
        countPipelineState.threadExecutionWidth, countThreadsPerGrid)

    countEncoder.dispatchThreads(
        MTLSize(width: countThreadsPerGrid, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: countThreadsPerThreadgroup, height: 1, depth: 1)
    )

    countEncoder.endEncoding()

    // Commit the command buffer and wait for it to complete
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Read the result
    let primeCount = countPtr.pointee

    let endTime = CFAbsoluteTimeGetCurrent()
    let timeElapsed = endTime - startTime

    print("Found \(primeCount) prime numbers in \(timeElapsed) seconds")

    // For verification on small ranges, you can extract and print the primes
    if range < 1000 {
        var primes: [UInt32] = []
        let sievePtr = sieveBuffer.contents().bindMemory(to: UInt32.self, capacity: sieveSize)

        for i in 0...range {
            if (sievePtr[Int(i) / 32] & (1 << (Int(i) % 32))) != 0 {
                primes.append(i)
            }
        }

        print("Primes: \(primes)")
    }
}

// For comparison: CPU implementation
func countPrimesSequential(range: UInt32) {
    print("Counting primes from 0 to \(range) using CPU...")
    let startTime = CFAbsoluteTimeGetCurrent()

    // Create the sieve array
    var sieve = [Bool](repeating: true, count: Int(range) + 1)
    sieve[0] = false
    sieve[1] = false

    // Apply the Sieve of Eratosthenes
    let sqrtRange = UInt32(sqrt(Float(range)))
    for i in 2...sqrtRange {
        if sieve[Int(i)] {
            var j = i * i
            while j <= range {
                sieve[Int(j)] = false
                j += i
            }
        }
    }

    // Count the primes
    let primeCount = sieve.filter { $0 }.count

    let endTime = CFAbsoluteTimeGetCurrent()
    let timeElapsed = endTime - startTime

    print("Found \(primeCount) prime numbers in \(timeElapsed) seconds")
}

// Execute both implementations for comparison
let range: UInt32 = 10_000_000  // Adjust as needed, using a smaller value for testing
countPrimesSequential(range: range)
countPrimesWithMetal(range: range)
