#pragma once

#include "Operator.h"
#include <vector>
#include <cstdint>

// Hash table structure for linear probing
// Uses structure-of-arrays layout for SIMD-friendly access
struct LinearProbingHashTable
{
    std::vector<int32_t> Keys;      // Hash table keys (EMPTY_KEY for empty slots)
    std::vector<int32_t> RowIndices; // Row index into build side data
    size_t Capacity;                 // Always power of 2
    size_t Mask;                     // Capacity - 1, for fast modulo
    size_t Count;                    // Number of entries

    static constexpr int32_t EMPTY_KEY = INT32_MIN;  // Sentinel for empty slots

    LinearProbingHashTable() : Capacity(0), Mask(0), Count(0) {}

    void Allocate(size_t numElements)
    {
        // Use ~50% load factor for good performance
        Capacity = 1;
        while (Capacity < numElements * 2)
            Capacity *= 2;

        Mask = Capacity - 1;
        Count = 0;

        Keys.resize(Capacity, EMPTY_KEY);
        RowIndices.resize(Capacity, -1);
    }

    void Clear()
    {
        std::fill(Keys.begin(), Keys.end(), EMPTY_KEY);
        std::fill(RowIndices.begin(), RowIndices.end(), -1);
        Count = 0;
    }
};

class HashJoinOperator : public BinaryOperator
{
public:
    // Left child = build side, Right child = probe side
    // buildKeyCol = column index for join key in build side
    // probeKeyCol = column index for join key in probe side
    HashJoinOperator(std::unique_ptr<Operator> buildSide,
                     std::unique_ptr<Operator> probeSide,
                     int buildKeyCol,
                     int probeKeyCol,
                     ExecutionMode mode);

    DataChunk Next() override;

private:
    int BuildKeyColumn;
    int ProbeKeyColumn;

    // Hash table
    LinearProbingHashTable HT;

    // Materialized build side data
    std::vector<DataChunk> BuildChunks;
    std::vector<int64_t> ChunkStartIndices;  // Global row index where each chunk starts
    int64_t TotalBuildRows;

    bool BuildPhaseComplete;

    // Probe state
    size_t CurrentProbeChunkIndex;
    std::vector<DataChunk> ProbeChunks;  // Buffer probe chunks for repeated iteration

    // Build phase
    void BuildHashTable();
    void ScalarBuild();
    void Avx2Build();

    // Probe phase - returns joined result chunk
    DataChunk ScalarProbe(const DataChunk& probeChunk);
    DataChunk Avx2Probe(const DataChunk& probeChunk);

    // Helper: get build row data by global row index
    std::pair<size_t, int64_t> GetBuildChunkAndLocalIndex(int64_t globalRowIndex) const;

    // Hash function: multiply-shift hash
    static inline uint32_t HashKey(int32_t key)
    {
        // Multiply-shift hash function (Dietzfelbinger et al.)
        // Good distribution and fast SIMD implementation
        constexpr uint32_t PRIME = 2654435769u;  // Golden ratio * 2^32
        return static_cast<uint32_t>(key) * PRIME;
    }
};
