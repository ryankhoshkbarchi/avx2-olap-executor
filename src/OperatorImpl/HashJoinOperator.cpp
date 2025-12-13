#include "pch.h"
#include "HashJoinOperator.h"
#include <arrow/builder.h>
#include <arrow/array.h>
#include "Misc/Logger.h"

HashJoinOperator::HashJoinOperator(
    std::unique_ptr<Operator> buildSide,
    std::unique_ptr<Operator> probeSide,
    int buildKeyCol,
    int probeKeyCol,
    ExecutionMode mode)
    : BinaryOperator(std::move(buildSide), std::move(probeSide), mode),
      BuildKeyColumn(buildKeyCol),
      ProbeKeyColumn(probeKeyCol),
      TotalBuildRows(0),
      BuildPhaseComplete(false),
      CurrentProbeChunkIndex(0)
{
}

DataChunk HashJoinOperator::Next()
{
    // Build phase: materialize build side and construct hash table
    if (!BuildPhaseComplete)
    {
        BuildHashTable();
        BuildPhaseComplete = true;
    }

    // Probe phase: process probe chunks
    while (true)
    {
        DataChunk probeChunk;

        // Get next probe chunk
        if (CurrentProbeChunkIndex < ProbeChunks.size())
        {
            probeChunk = ProbeChunks[CurrentProbeChunkIndex++];
        }
        else
        {
            probeChunk = RightChild->Next();
            if (probeChunk == nullptr)
            {
                return nullptr;  // Done
            }
            ProbeChunks.push_back(probeChunk);
            CurrentProbeChunkIndex++;
        }

        // Probe the hash table
        DataChunk result;
        if (CurrentMode == ExecutionMode::AVX2)
        {
            result = Avx2Probe(probeChunk);
        }
        else
        {
            result = ScalarProbe(probeChunk);
        }

        // Return result if it has rows (some probe chunks may have no matches)
        if (result != nullptr && result->num_rows() > 0)
        {
            return result;
        }
    }
}

void HashJoinOperator::BuildHashTable()
{
    // Materialize build side
    DataChunk chunk;
    while ((chunk = LeftChild->Next()) != nullptr)
    {
        ChunkStartIndices.push_back(TotalBuildRows);
        TotalBuildRows += chunk->num_rows();
        BuildChunks.push_back(chunk);
    }

    if (TotalBuildRows == 0)
    {
        return;
    }

    // Allocate hash table
    HT.Allocate(static_cast<size_t>(TotalBuildRows));

    // Build using appropriate method
    if (CurrentMode == ExecutionMode::AVX2)
    {
        Avx2Build();
    }
    else
    {
        ScalarBuild();
    }
}

std::pair<size_t, int64_t> HashJoinOperator::GetBuildChunkAndLocalIndex(int64_t globalRowIndex) const
{
    // Binary search to find which chunk contains this row
    size_t chunkIdx = 0;
    for (size_t i = 0; i < ChunkStartIndices.size(); i++)
    {
        if (i + 1 < ChunkStartIndices.size() && ChunkStartIndices[i + 1] <= globalRowIndex)
        {
            continue;
        }
        chunkIdx = i;
        break;
    }
    int64_t localIdx = globalRowIndex - ChunkStartIndices[chunkIdx];
    return {chunkIdx, localIdx};
}

// ============================================================================
// SCALAR IMPLEMENTATION
// ============================================================================

void HashJoinOperator::ScalarBuild()
{
    int64_t globalRowIdx = 0;

    for (const auto& chunk : BuildChunks)
    {
        auto keyColumn = std::static_pointer_cast<arrow::Int32Array>(
            chunk->column(BuildKeyColumn));

        const int32_t* keys = keyColumn->raw_values();
        int64_t numRows = chunk->num_rows();

        for (int64_t i = 0; i < numRows; i++)
        {
            int32_t key = keys[i];
            uint32_t hash = HashKey(key);
            size_t pos = hash & HT.Mask;

            // Linear probing to find empty slot
            while (HT.Keys[pos] != LinearProbingHashTable::EMPTY_KEY)
            {
                pos = (pos + 1) & HT.Mask;
            }

            HT.Keys[pos] = key;
            HT.RowIndices[pos] = static_cast<int32_t>(globalRowIdx + i);
            HT.Count++;
        }

        globalRowIdx += numRows;
    }
}

DataChunk HashJoinOperator::ScalarProbe(const DataChunk& probeChunk)
{
    auto probeKeyColumn = std::static_pointer_cast<arrow::Int32Array>(
        probeChunk->column(ProbeKeyColumn));

    const int32_t* probeKeys = probeKeyColumn->raw_values();
    int64_t numProbeRows = probeChunk->num_rows();

    // Collect matching pairs: (probe row index, build global row index)
    std::vector<int64_t> probeMatches;
    std::vector<int64_t> buildMatches;

    for (int64_t i = 0; i < numProbeRows; i++)
    {
        int32_t probeKey = probeKeys[i];
        uint32_t hash = HashKey(probeKey);
        size_t pos = hash & HT.Mask;

        // Linear probing to find key
        while (HT.Keys[pos] != LinearProbingHashTable::EMPTY_KEY)
        {
            if (HT.Keys[pos] == probeKey)
            {
                probeMatches.push_back(i);
                buildMatches.push_back(HT.RowIndices[pos]);
                // Note: For inner join with unique keys, we could break here
                // But for general case with duplicates, continue searching
            }
            pos = (pos + 1) & HT.Mask;

            // Safety check: if we've wrapped around completely, key not found
            if (pos == (hash & HT.Mask))
                break;
        }
    }

    if (probeMatches.empty())
    {
        return nullptr;
    }

    // Build output: concatenate columns from both sides
    // Output schema: [probe columns..., build columns...]
    std::vector<std::shared_ptr<arrow::Array>> outputArrays;

    // First, add probe side columns
    for (int col = 0; col < probeChunk->num_columns(); col++)
    {
        auto probeArray = probeChunk->column(col);

        // For simplicity, assume int32 columns
        auto typedArray = std::static_pointer_cast<arrow::Int32Array>(probeArray);
        arrow::Int32Builder builder;

        for (size_t m = 0; m < probeMatches.size(); m++)
        {
            auto status = builder.Append(typedArray->Value(probeMatches[m]));
            (void)status;
        }

        std::shared_ptr<arrow::Array> result;
        auto status = builder.Finish(&result);
        (void)status;
        outputArrays.push_back(result);
    }

    // Then, add build side columns
    if (!BuildChunks.empty())
    {
        int numBuildCols = BuildChunks[0]->num_columns();
        for (int col = 0; col < numBuildCols; col++)
        {
            arrow::Int32Builder builder;

            for (size_t m = 0; m < buildMatches.size(); m++)
            {
                auto [chunkIdx, localIdx] = GetBuildChunkAndLocalIndex(buildMatches[m]);
                auto buildArray = std::static_pointer_cast<arrow::Int32Array>(
                    BuildChunks[chunkIdx]->column(col));
                auto status = builder.Append(buildArray->Value(localIdx));
                (void)status;
            }

            std::shared_ptr<arrow::Array> result;
            auto status = builder.Finish(&result);
            (void)status;
            outputArrays.push_back(result);
        }
    }

    // Create output schema
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int i = 0; i < probeChunk->num_columns(); i++)
    {
        fields.push_back(arrow::field("probe_" + std::to_string(i), arrow::int32()));
    }
    if (!BuildChunks.empty())
    {
        for (int i = 0; i < BuildChunks[0]->num_columns(); i++)
        {
            fields.push_back(arrow::field("build_" + std::to_string(i), arrow::int32()));
        }
    }

    auto schema = arrow::schema(fields);
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(probeMatches.size()), outputArrays);
}

// ============================================================================
// AVX2 SIMD IMPLEMENTATION
// ============================================================================

void HashJoinOperator::Avx2Build()
{
    // Constants for SIMD hash computation
    const __m256i vPrime = _mm256_set1_epi32(2654435769u);
    const __m256i vMask = _mm256_set1_epi32(static_cast<int32_t>(HT.Mask));
    const __m256i vEmpty = _mm256_set1_epi32(LinearProbingHashTable::EMPTY_KEY);
    const __m256i vOne = _mm256_set1_epi32(1);

    int64_t globalRowIdx = 0;

    for (const auto& chunk : BuildChunks)
    {
        auto keyColumn = std::static_pointer_cast<arrow::Int32Array>(
            chunk->column(BuildKeyColumn));

        const int32_t* keys = keyColumn->raw_values();
        int64_t numRows = chunk->num_rows();
        int64_t i = 0;

        // Process 8 keys at a time
        for (; i + 8 <= numRows; i += 8)
        {
            // Load 8 keys
            __m256i vKeys = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));

            // Compute hash: (key * PRIME) & mask
            __m256i vHash = _mm256_mullo_epi32(vKeys, vPrime);
            __m256i vPos = _mm256_and_si256(vHash, vMask);

            // Extract positions and insert one by one (AVX2 doesn't have scatter)
            alignas(32) int32_t positions[8];
            alignas(32) int32_t keyValues[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(positions), vPos);
            _mm256_store_si256(reinterpret_cast<__m256i*>(keyValues), vKeys);

            for (int lane = 0; lane < 8; lane++)
            {
                int32_t key = keyValues[lane];
                size_t pos = static_cast<size_t>(positions[lane]);

                // Linear probing to find empty slot
                while (HT.Keys[pos] != LinearProbingHashTable::EMPTY_KEY)
                {
                    pos = (pos + 1) & HT.Mask;
                }

                HT.Keys[pos] = key;
                HT.RowIndices[pos] = static_cast<int32_t>(globalRowIdx + i + lane);
                HT.Count++;
            }
        }

        // Handle remaining rows with scalar code
        for (; i < numRows; i++)
        {
            int32_t key = keys[i];
            uint32_t hash = HashKey(key);
            size_t pos = hash & HT.Mask;

            while (HT.Keys[pos] != LinearProbingHashTable::EMPTY_KEY)
            {
                pos = (pos + 1) & HT.Mask;
            }

            HT.Keys[pos] = key;
            HT.RowIndices[pos] = static_cast<int32_t>(globalRowIdx + i);
            HT.Count++;
        }

        globalRowIdx += numRows;
    }
}

DataChunk HashJoinOperator::Avx2Probe(const DataChunk& probeChunk)
{
    auto probeKeyColumn = std::static_pointer_cast<arrow::Int32Array>(
        probeChunk->column(ProbeKeyColumn));

    const int32_t* probeKeys = probeKeyColumn->raw_values();
    int64_t numProbeRows = probeChunk->num_rows();

    // Collect matching pairs
    std::vector<int64_t> probeMatches;
    std::vector<int64_t> buildMatches;

    // SIMD constants
    const __m256i vPrime = _mm256_set1_epi32(2654435769u);
    const __m256i vMask = _mm256_set1_epi32(static_cast<int32_t>(HT.Mask));
    const __m256i vEmpty = _mm256_set1_epi32(LinearProbingHashTable::EMPTY_KEY);
    const __m256i vOne = _mm256_set1_epi32(1);

    int64_t i = 0;

    // Process 8 probe keys at a time using vectorized probing
    for (; i + 8 <= numProbeRows; i += 8)
    {
        // Load 8 probe keys
        __m256i vProbeKeys = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&probeKeys[i]));

        // Compute initial hash positions
        __m256i vHash = _mm256_mullo_epi32(vProbeKeys, vPrime);
        __m256i vPos = _mm256_and_si256(vHash, vMask);

        // Track which lanes are still active (haven't found match or empty)
        uint32_t activeMask = 0xFF;  // All 8 lanes active

        // Store positions for iteration
        alignas(32) int32_t positions[8];
        alignas(32) int32_t probeKeyValues[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(positions), vPos);
        _mm256_store_si256(reinterpret_cast<__m256i*>(probeKeyValues), vProbeKeys);

        // Linear probing with SIMD gather
        const int maxProbes = static_cast<int>(HT.Capacity);  // Safety limit
        for (int probe = 0; probe < maxProbes && activeMask != 0; probe++)
        {
            // Gather keys from hash table at current positions
            __m256i vHTKeys = _mm256_i32gather_epi32(HT.Keys.data(), vPos, 4);

            // Check for matches: probeKey == htKey
            __m256i vMatchCmp = _mm256_cmpeq_epi32(vProbeKeys, vHTKeys);
            uint32_t matchMask = static_cast<uint32_t>(_mm256_movemask_ps(
                _mm256_castsi256_ps(vMatchCmp)));

            // Check for empty slots: htKey == EMPTY_KEY
            __m256i vEmptyCmp = _mm256_cmpeq_epi32(vHTKeys, vEmpty);
            uint32_t emptyMask = static_cast<uint32_t>(_mm256_movemask_ps(
                _mm256_castsi256_ps(vEmptyCmp)));

            // Process matches
            uint32_t foundMatches = matchMask & activeMask;
            while (foundMatches)
            {
                // Find first set bit
                int lane = __builtin_ctz(foundMatches);

                // Record match
                probeMatches.push_back(i + lane);
                buildMatches.push_back(HT.RowIndices[positions[lane]]);

                // Clear this bit
                foundMatches &= ~(1u << lane);

                // Deactivate this lane (found match)
                activeMask &= ~(1u << lane);
            }

            // Deactivate lanes that hit empty slots (no match exists)
            activeMask &= ~emptyMask;

            if (activeMask == 0)
                break;

            // Advance positions for still-active lanes (linear probing)
            vPos = _mm256_add_epi32(vPos, vOne);
            vPos = _mm256_and_si256(vPos, vMask);

            // Update positions array for active lanes
            _mm256_store_si256(reinterpret_cast<__m256i*>(positions), vPos);
        }
    }

    // Handle remaining rows with scalar code
    for (; i < numProbeRows; i++)
    {
        int32_t probeKey = probeKeys[i];
        uint32_t hash = HashKey(probeKey);
        size_t pos = hash & HT.Mask;

        while (HT.Keys[pos] != LinearProbingHashTable::EMPTY_KEY)
        {
            if (HT.Keys[pos] == probeKey)
            {
                probeMatches.push_back(i);
                buildMatches.push_back(HT.RowIndices[pos]);
            }
            pos = (pos + 1) & HT.Mask;
            if (pos == (hash & HT.Mask))
                break;
        }
    }

    if (probeMatches.empty())
    {
        return nullptr;
    }

    // Build output (same as scalar version)
    std::vector<std::shared_ptr<arrow::Array>> outputArrays;

    // Add probe side columns
    for (int col = 0; col < probeChunk->num_columns(); col++)
    {
        auto probeArray = probeChunk->column(col);
        auto typedArray = std::static_pointer_cast<arrow::Int32Array>(probeArray);
        arrow::Int32Builder builder;

        for (size_t m = 0; m < probeMatches.size(); m++)
        {
            auto status = builder.Append(typedArray->Value(probeMatches[m]));
            (void)status;
        }

        std::shared_ptr<arrow::Array> result;
        auto status = builder.Finish(&result);
        (void)status;
        outputArrays.push_back(result);
    }

    // Add build side columns
    if (!BuildChunks.empty())
    {
        int numBuildCols = BuildChunks[0]->num_columns();
        for (int col = 0; col < numBuildCols; col++)
        {
            arrow::Int32Builder builder;

            for (size_t m = 0; m < buildMatches.size(); m++)
            {
                auto [chunkIdx, localIdx] = GetBuildChunkAndLocalIndex(buildMatches[m]);
                auto buildArray = std::static_pointer_cast<arrow::Int32Array>(
                    BuildChunks[chunkIdx]->column(col));
                auto status = builder.Append(buildArray->Value(localIdx));
                (void)status;
            }

            std::shared_ptr<arrow::Array> result;
            auto status = builder.Finish(&result);
            (void)status;
            outputArrays.push_back(result);
        }
    }

    // Create output schema
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int i = 0; i < probeChunk->num_columns(); i++)
    {
        fields.push_back(arrow::field("probe_" + std::to_string(i), arrow::int32()));
    }
    if (!BuildChunks.empty())
    {
        for (int i = 0; i < BuildChunks[0]->num_columns(); i++)
        {
            fields.push_back(arrow::field("build_" + std::to_string(i), arrow::int32()));
        }
    }

    auto schema = arrow::schema(fields);
    return arrow::RecordBatch::Make(schema, static_cast<int64_t>(probeMatches.size()), outputArrays);
}
