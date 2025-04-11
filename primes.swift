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
            return (sieve[index / 32] & (1 << (index % 32))) != 0;
        }

        inline void clearBit(device uint* sieve, uint index) {
            atomic_fetch_and_explicit(
                (device atomic_uint*)(sieve + (index / 32)),
                ~(1 << (index % 32)),
                memory_order_relaxed
            );
        }

        // Initialize the sieve
        kernel void initSieve(
            device uint* sieve [[buffer(0)]],
            constant uint& range [[buffer(1)]],
            uint id [[thread_position_in_grid]],
            uint threads [[threads_per_grid]])
        {
            uint elementsPerThread = ((range / 32) + threads - 1) / threads;
            uint start = id * elementsPerThread;
            uint end = min(start + elementsPerThread, (range / 32) + 1);

            // Initialize all bits to 1 (prime)
            for (uint i = start; i < end; i++) {
                sieve[i] = 0xFFFFFFFF;
            }

            // Ensure thread 0 marks 0 and 1 as non-prime
            if (id == 0) {
                sieve[0] &= ~(1u << 0); // 0 is not prime
                sieve[0] &= ~(1u << 1); // 1 is not prime
            }
        }

        // Mark multiples as non-prime
        kernel void markMultiples(
            device uint* sieve [[buffer(0)]],
            constant uint& range [[buffer(1)]],
            uint id [[thread_position_in_grid]],
            uint threads [[threads_per_grid]])
        {
            // Skip thread IDs 0 and 1
            if (id <= 1) return;

            // Only process if this ID is still marked prime
            uint blockIdx = id / 32;
            uint bitIdx = id % 32;

            bool isPrime = (sieve[blockIdx] & (1 << bitIdx)) != 0;
            uint sqrtRange = uint(sqrt(float(range)));

            // Only threads representing prime numbers <= sqrt(range) mark multiples
            if (isPrime && id <= sqrtRange) {
                // Start from id*id and mark all multiples as non-prime
                for (uint j = id * id; j <= range; j += id) {
                    clearBit(sieve, j);
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

    // Dispatch with optimal thread count for initialization
    let initThreads = min(initPipeline.maxTotalThreadsPerThreadgroup, sieveSize)
    initEncoder.dispatchThreads(
        MTLSize(width: initThreads, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: initThreads, height: 1, depth: 1)
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

    // Use enough threads to cover all potential primes up to range
    // Each thread ID is a potential prime number
    let sqrtRange = Int(sqrt(Float(range)))
    let markThreads = min(markPipeline.maxTotalThreadsPerThreadgroup, sqrtRange + 1)
    let markThreadgroups = (sqrtRange + markThreads) / markThreads

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
let range: UInt32 = 100_000_000  // 100 million
countPrimesWithMetalAccurate(range: range)
