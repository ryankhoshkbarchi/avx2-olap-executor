#pragma once

#include "Operator.h"
#include <vector>
#include <cstdint>

// Cuckoo hash table structure (Polychroniou et al.)
// Uses two hash functions - each key can be in one of two positions
// SIMD-friendly: can check both positions in parallel during probe
struct CuckooHashTable
{
    std::vector<int32_t> Keys;       // Hash table keys (EMPTY_KEY for empty slots)
    std::vector<int32_t> RowIndices; // Row index into build side data
    size_t Capacity;                  // Size of each table (total size = 2 * Capacity)
    size_t Mask;                      // Capacity - 1, for fast modulo

    static constexpr int32_t EMPTY_KEY = INT32_MIN;  // Sentinel for empty slots
    static constexpr int MAX_KICKS = 500;            // Max displacement chain length

    // Two hash functions using different primes
    static constexpr uint32_t PRIME1 = 2654435769u;  // Golden ratio * 2^32
    static constexpr uint32_t PRIME2 = 2246822519u;  // Another prime

    CuckooHashTable() : Capacity(0), Mask(0) {}

    void Allocate(size_t numElements)
    {
        // Cuckoo hashing needs ~55% load factor for reliable insertion
        Capacity = 1;
        while (Capacity < numElements * 2)
            Capacity *= 2;

        Mask = Capacity - 1;

        // Single array: first half for hash1, second half for hash2
        Keys.resize(Capacity * 2, EMPTY_KEY);
        RowIndices.resize(Capacity * 2, -1);
    }

    // Hash function 1: position in first table [0, Capacity)
    inline size_t Hash1(int32_t key) const
    {
        return (static_cast<uint32_t>(key) * PRIME1) & Mask;
    }

    // Hash function 2: position in second table [Capacity, 2*Capacity)
    inline size_t Hash2(int32_t key) const
    {
        return Capacity + ((static_cast<uint32_t>(key) * PRIME2) & Mask);
    }

    void Clear()
    {
        std::fill(Keys.begin(), Keys.end(), EMPTY_KEY);
        std::fill(RowIndices.begin(), RowIndices.end(), -1);
    }
};

class CuckooHashJoinOperator : public BinaryOperator
{
public:
    // Left child = build side, Right child = probe side
    CuckooHashJoinOperator(std::unique_ptr<Operator> buildSide,
                           std::unique_ptr<Operator> probeSide,
                           int buildKeyCol,
                           int probeKeyCol,
                           ExecutionMode mode);

    DataChunk Next() override;

private:
    int BuildKeyColumn;
    int ProbeKeyColumn;

    // Cuckoo hash table
    CuckooHashTable HT;

    // Materialized build side data
    std::vector<DataChunk> BuildChunks;
    std::vector<int64_t> ChunkStartIndices;
    int64_t TotalBuildRows;

    bool BuildPhaseComplete;

    // Probe state
    size_t CurrentProbeChunkIndex;
    std::vector<DataChunk> ProbeChunks;

    // Build phase
    void BuildHashTable();
    bool ScalarInsert(int32_t key, int32_t rowIndex);

    // Probe phase
    DataChunk ScalarProbe(const DataChunk& probeChunk);
    DataChunk Avx2Probe(const DataChunk& probeChunk);

    // Helper: get build row data by global row index
    std::pair<size_t, int64_t> GetBuildChunkAndLocalIndex(int64_t globalRowIndex) const;
};
