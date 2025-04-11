import Foundation
import Metal

func countPrimesWithMetalAccurate(range: UInt32) {
    print("Counting primes from 0 to \(range) using accurate Metal GPU implementation...")
    let startTime = CFAbsoluteTimeGetCurrent()

    // Create the Metal device and command queue
    guard let device = MTLCreateSystemDefaultDevice(),
        let commandQueue = device.makeCommandQueue()
    else {
        fatalError("Metal setup failed")
    }

    // Define a more accurate Metal shader
    let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;

            // Simple bit operations for bit-packed sieve
            inline bool getBit(device const uint* sieve, uint index) {
                return (sieve[index / 32] & (1u << (index % 32))) != 0;
            }

            inline void clearBit(device uint* sieve, uint index) {
                atomic_fetch_and_explicit(
                    (device atomic_uint*)(sieve + (index / 32)),
                    ~(1u << (index % 32)),
                    memory_order_relaxed
                );
            }

            // Initialize the sieve
            kernel void initSieve(
                device uint* sieve [[buffer(0)]],
                constant uint& range [[buffer(1)]],
                uint id [[thread_position_in_grid]])
            {
                // Fill with 1s (all assumed prime initially)
                if (id < (range / 32 + 1)) {
                    sieve[id] = 0xFFFFFFFF;
                }

                // Mark 0 and 1 as non-prime
                if (id == 0) {
                    sieve[0] &= ~(1u); // Clear bit 0
                    sieve[0] &= ~(2u); // Clear bit 1
                }
            }

            // Mark multiples as non-prime using Sieve of Eratosthenes
            kernel void markMultiples(
                device uint* sieve [[buffer(0)]],
                constant uint& range [[buffer(1)]],
                uint id [[thread_position_in_grid]],
                uint threads [[threads_per_grid]])
            {
                // Each thread processes a chunk of starting primes
                uint sqrtRange = uint(sqrt(float(range)));
                uint primesPerThread = (sqrtRange + threads - 1) / threads;
                uint startPrime = 2 + id * primesPerThread;
                uint endPrime = min(startPrime + primesPerThread, sqrtRange + 1);

                for (uint p = startPrime; p < endPrime; p++) {
                    // Check if p is still marked as prime
                    if (getBit(sieve, p)) {
                        // Mark all multiples of p (starting from p*p) as non-prime
                        for (uint multiple = p * p; multiple <= range; multiple += p) {
                            clearBit(sieve, multiple);
                        }
                    }
                }
            }

            // Count the primes in the sieve
            kernel void countPrimes(
                device const uint* sieve [[buffer(0)]],
                constant uint& range [[buffer(1)]],
                device atomic_uint* totalCount [[buffer(2)]],
                uint id [[thread_position_in_grid]],
                uint threads [[threads_per_grid]])
            {
                // Each thread counts primes in its own chunk
                uint numbersPerThread = (range + threads - 1) / threads;
                uint start = id * numbersPerThread;
                uint end = min(start + numbersPerThread, range + 1);

                uint localCount = 0;
                for (uint i = start; i < end; i++) {
                    if (getBit(sieve, i)) {
                        localCount++;
                    }
                }

                // Add local count to global count
                atomic_fetch_add_explicit(totalCount, localCount, memory_order_relaxed);
            }
        """

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: shaderSource, options: nil)
    } catch {
        fatalError("Failed to create Metal library: \(error)")
    }

    // Calculate buffer size (bit-packed: 32 numbers per uint)
    let sieveSize = (Int(range) / 32) + 1
    let sieveBufferSize = sieveSize * MemoryLayout<UInt32>.size

    // Create buffers
    let sieveBuffer = device.makeBuffer(length: sieveBufferSize, options: .storageModeShared)!
    let rangeBuffer = device.makeBuffer(
        bytes: [range], length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    let countBuffer = device.makeBuffer(
        length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

    // Zero out the count
    let countPtr = countBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
    countPtr.pointee = 0

    // STEP 1: Initialize the sieve
    let initFunction = library.makeFunction(name: "initSieve")!
    let initPipeline = try! device.makeComputePipelineState(function: initFunction)

    let initCommandBuffer = commandQueue.makeCommandBuffer()!
    let initEncoder = initCommandBuffer.makeComputeCommandEncoder()!
    initEncoder.setComputePipelineState(initPipeline)
    initEncoder.setBuffer(sieveBuffer, offset: 0, index: 0)
    initEncoder.setBuffer(rangeBuffer, offset: 0, index: 1)

    // Dispatch enough threads to initialize the entire sieve
    let threadgroupSize = min(initPipeline.maxTotalThreadsPerThreadgroup, 256)
    let threadgroups = (sieveSize + threadgroupSize - 1) / threadgroupSize

    initEncoder.dispatchThreadgroups(
        MTLSize(width: threadgroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
    )
    initEncoder.endEncoding()
    initCommandBuffer.commit()
    initCommandBuffer.waitUntilCompleted()

    // STEP 2: Mark multiples
    let markFunction = library.makeFunction(name: "markMultiples")!
    let markPipeline = try! device.makeComputePipelineState(function: markFunction)

    let markCommandBuffer = commandQueue.makeCommandBuffer()!
    let markEncoder = markCommandBuffer.makeComputeCommandEncoder()!
    markEncoder.setComputePipelineState(markPipeline)
    markEncoder.setBuffer(sieveBuffer, offset: 0, index: 0)
    markEncoder.setBuffer(rangeBuffer, offset: 0, index: 1)

    // Use a moderate number of threads for good parallelism
    let markThreads = min(markPipeline.maxTotalThreadsPerThreadgroup, 256)
    let markThreadgroups = 64  // Use enough threads to efficiently distribute work

    markEncoder.dispatchThreadgroups(
        MTLSize(width: markThreadgroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: markThreads, height: 1, depth: 1)
    )
    markEncoder.endEncoding()
    markCommandBuffer.commit()
    markCommandBuffer.waitUntilCompleted()

    // STEP 3: Count the primes
    let countFunction = library.makeFunction(name: "countPrimes")!
    let countPipeline = try! device.makeComputePipelineState(function: countFunction)

    let countCommandBuffer = commandQueue.makeCommandBuffer()!
    let countEncoder = countCommandBuffer.makeComputeCommandEncoder()!
    countEncoder.setComputePipelineState(countPipeline)
    countEncoder.setBuffer(sieveBuffer, offset: 0, index: 0)
    countEncoder.setBuffer(rangeBuffer, offset: 0, index: 1)
    countEncoder.setBuffer(countBuffer, offset: 0, index: 2)

    // Use optimal thread count for counting
    let countThreads = min(countPipeline.maxTotalThreadsPerThreadgroup, 512)
    let countThreadgroups = 64  // Use more threads for better parallelism during counting

    countEncoder.dispatchThreadgroups(
        MTLSize(width: countThreadgroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: countThreads, height: 1, depth: 1)
    )
    countEncoder.endEncoding()
    countCommandBuffer.commit()
    countCommandBuffer.waitUntilCompleted()

    // Read the result
    let primeCount = countPtr.pointee

    let endTime = CFAbsoluteTimeGetCurrent()
    let timeElapsed = endTime - startTime

    print(
        "Found \(primeCount) prime numbers in \(timeElapsed) seconds using accurate GPU implementation"
    )

    // Verify small range results if needed
    if range < 1000 {
        var primes: [UInt32] = []
        let sievePtr = sieveBuffer.contents().bindMemory(to: UInt32.self, capacity: sieveSize)

        for i in 0...range {
            if (sievePtr[Int(i) / 32] & (1 << (Int(i) % 32))) != 0 {
                primes.append(i)
            }
        }

        print("First few primes: \(primes.prefix(20))...")
    }
}

// Execute both implementations
let range: UInt32 = 1_000_000_000
countPrimesWithMetalAccurate(range: range)
