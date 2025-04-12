import Foundation
import Metal

func countPrimesWithMetalHighlyOptimized(range: UInt32) {
    print("Counting primes from 0 to \(range) using highly optimized Metal GPU implementation...")
    let startTime = CFAbsoluteTimeGetCurrent()

    guard let device = MTLCreateSystemDefaultDevice(),
        let commandQueue = device.makeCommandQueue()
    else {
        fatalError("Metal setup failed")
    }

    // Highly optimized shader using bit-level operations and wheel factorization
    let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        #define WHEEL_SIZE 30
        #define WHEEL_PRIMES 3  // 2, 3, 5

        // Optimized bit operations with direct indexing
        inline bool getBit(device const uint* sieve, uint index) {
            return (sieve[index >> 5] & (1u << (index & 31))) != 0;
        }

        inline void clearBit(device uint* sieve, uint index) {
            atomic_fetch_and_explicit(
                (device atomic_uint*)(sieve + (index >> 5)),
                ~(1u << (index & 31)),
                memory_order_relaxed
            );
        }

        // Initialize the sieve with wheel pattern to skip multiples of 2, 3, and 5
        kernel void initSieve(
            device uint* sieve [[buffer(0)]],
            constant uint& range [[buffer(1)]],
            uint id [[thread_position_in_grid]],
            uint threads [[threads_per_grid]])
        {
            // Calculate range each thread handles
            uint elemsPerThread = (range / 32 + threads - 1) / threads;
            uint startElem = id * elemsPerThread;
            uint endElem = min(startElem + elemsPerThread, (range / 32 + 1));

            // Set all bits to 1 (assuming all prime initially)
            for (uint i = startElem; i < endElem; i++) {
                sieve[i] = 0xFFFFFFFF;
            }

            if (id == 0) {
                // Mark 0 and 1 as non-prime
                sieve[0] &= ~(1u | 2u);

                // Keep 2, 3, 5 as prime and mark their multiples
                // We'll skip 2, 3, 5 in the wheel pattern
                for (uint i = 4; i <= min(10u, range); i++) {
                    if (i != 2 && i != 3 && i != 5) {
                        clearBit(sieve, i);
                    }
                }
            }

            // Wheel pattern: Only values coprime to 2,3,5 can be prime
            // The wheel pattern after 30 repeats, so we can use modular arithmetic
            const uchar wheel[WHEEL_SIZE] = {
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 0, 0, 1 // 1 means potentially prime, 0 means composite
            };

            // Number of values to check in each thread
            uint valuesPerThread = (range + threads - 1) / threads;
            uint startVal = id * valuesPerThread;
            uint endVal = min(startVal + valuesPerThread, range + 1);

            // Skip first few special cases
            startVal = max(startVal, 11u);

            // Apply wheel pattern to mark non-primes
            for (uint i = startVal; i < endVal; i++) {
                if (wheel[i % WHEEL_SIZE] == 0) {
                    clearBit(sieve, i);
                }
            }
        }

        // Mark multiples as non-prime using segmented sieve approach
        kernel void markMultiples(
            device uint* sieve [[buffer(0)]],
            constant uint& range [[buffer(1)]],
            uint id [[thread_position_in_grid]],
            uint threads [[threads_per_grid]])
        {
            uint sqrtRange = uint(sqrt(float(range)));

            // Wheel pattern after removing multiples of 2, 3, 5
            // These are the indices of numbers coprime to 2,3,5
            const uchar wheelIndices[8] = {1, 7, 11, 13, 17, 19, 23, 29};
            const uint numIndices = 8;

            // Each thread handles a range of wheel patterns
            uint patternsPerThread = (sqrtRange / WHEEL_SIZE + threads - 1) / threads;
            uint startPattern = id * patternsPerThread;
            uint endPattern = min(startPattern + patternsPerThread, sqrtRange / WHEEL_SIZE + 1);

            // Process all wheel patterns assigned to this thread
            for (uint pattern = startPattern; pattern < endPattern; pattern++) {
                uint wheelBase = pattern * WHEEL_SIZE;

                // Process each potential prime in the wheel
                for (uint i = 0; i < numIndices; i++) {
                    uint prime = wheelBase + wheelIndices[i];

                    if (prime <= 10 || prime > sqrtRange) continue; // Skip small primes and beyond sqrt

                    // Check if this is still marked as prime
                    if (getBit(sieve, prime)) {
                        // Calculate squared value (starting point for marking)
                        uint primeSquared = prime * prime;

                        // For each prime, mark all its multiples
                        // Process 16KB segments at a time for better cache performance
                        uint segSize = 16384;

                        for (uint segStart = primeSquared; segStart <= range; segStart += segSize) {
                            uint segEnd = min(segStart + segSize, range + 1);

                            // Mark multiples in this segment
                            for (uint j = 0; j < numIndices; j++) {
                                // Calculate the first multiple of prime in this segment
                                uint pattern = (wheelBase + wheelIndices[j]) % WHEEL_SIZE;
                                uint multiple = prime * pattern;

                                // Ensure we're at or past the segment start
                                if (multiple < segStart) {
                                    uint offset = (segStart - multiple) % prime;
                                    if (offset > 0) offset = prime - offset;
                                    multiple = segStart + offset;
                                }

                                // Mark all multiples in this segment
                                for (; multiple < segEnd; multiple += prime) {
                                    if (multiple >= primeSquared) {
                                        clearBit(sieve, multiple);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Count primes using GPU-optimized parallel reduction
        kernel void countPrimes(
            device const uint* sieve [[buffer(0)]],
            constant uint& range [[buffer(1)]],
            device atomic_uint* totalCount [[buffer(2)]],
            threadgroup uint* localCounts [[threadgroup(0)]],
            uint id [[thread_position_in_grid]],
            uint localId [[thread_index_in_threadgroup]],
            uint threadsPerGroup [[threads_per_threadgroup]])
        {
            // Pre-count the first few primes
            uint specialPrimes = 0;
            if (id == 0) {
                specialPrimes = 3; // 2, 3, 5
            }

            // Process data in chunks for better cache efficiency
            uint elemsPerThread = (range / 32 + 1024 - 1) / 1024;

            uint localCount = 0;

            // Each thread counts bits in its assigned elements
            for (uint chunk = 0; chunk < elemsPerThread; chunk++) {
                uint elemIdx = id + chunk * 1024;
                if (elemIdx > range / 32) break;

                uint bits = sieve[elemIdx];

                // Fast bit counting with popcnt
                #if __METAL_VERSION__ >= 220
                    localCount += popcount(bits);
                #else
                    // Fallback for older Metal versions
                    bits = bits - ((bits >> 1) & 0x55555555);
                    bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333);
                    bits = (bits + (bits >> 4)) & 0x0F0F0F0F;
                    bits = bits + (bits >> 8);
                    bits = bits + (bits >> 16);
                    localCount += bits & 0x3F;
                #endif
            }

            // Store in local memory for reduction
            localCounts[localId] = localCount;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Parallel reduction within threadgroup
            for (uint stride = threadsPerGroup/2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localCounts[localId] += localCounts[localId + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Only thread 0 in each group adds to global counter
            if (localId == 0) {
                atomic_fetch_add_explicit(totalCount, localCounts[0] + (id == 0 ? specialPrimes : 0), memory_order_relaxed);
            }
        }
        """

    let compileOptions = MTLCompileOptions()
    compileOptions.optimizationLevel = .fast
    compileOptions.fastMathEnabled = true

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: shaderSource, options: compileOptions)
    } catch {
        fatalError("Failed to create Metal library: \(error)")
    }

    // Calculate buffer size
    let sieveSize = (Int(range) / 32) + 1
    let sieveBufferSize = sieveSize * MemoryLayout<UInt32>.size

    // Create buffers with optimal options
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

    // Use optimal thread configuration
    let initThreads = 256
    let initThreadgroups = 512

    initEncoder.dispatchThreadgroups(
        MTLSize(width: initThreadgroups, height: 1, depth: 1),
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

    // Use optimal thread configuration
    let markThreads = 256
    let markThreadgroups = 512

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

    let countThreads = 256
    let countThreadgroups = 1024

    // Create and set up command buffer
    let countCommandBuffer = commandQueue.makeCommandBuffer()!
    let countEncoder = countCommandBuffer.makeComputeCommandEncoder()!
    countEncoder.setComputePipelineState(countPipeline)
    countEncoder.setBuffer(sieveBuffer, offset: 0, index: 0)
    countEncoder.setBuffer(rangeBuffer, offset: 0, index: 1)
    countEncoder.setBuffer(countBuffer, offset: 0, index: 2)

    // Set up threadgroup memory for parallel reduction
    let threadsPerGroup = countThreads
    let localMemSize = threadsPerGroup * MemoryLayout<UInt32>.size
    countEncoder.setThreadgroupMemoryLength(localMemSize, index: 0)

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
        "Found \(primeCount) prime numbers in \(timeElapsed) seconds using highly optimized GPU implementation"
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

// Execute the highly optimized implementation
let range: UInt32 = 1_000_000_000  // 1 billion
countPrimesWithMetalHighlyOptimized(range: range)
