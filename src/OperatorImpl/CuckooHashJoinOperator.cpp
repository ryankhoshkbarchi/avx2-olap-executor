#include "pch.h"
#include "CuckooHashJoinOperator.h"
#include <arrow/builder.h>
#include <arrow/array.h>
#include "Misc/Logger.h"

CuckooHashJoinOperator::CuckooHashJoinOperator(
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

DataChunk CuckooHashJoinOperator::Next()
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

        if (CurrentProbeChunkIndex < ProbeChunks.size())
        {
            probeChunk = ProbeChunks[CurrentProbeChunkIndex++];
        }
        else
        {
            probeChunk = RightChild->Next();
            if (probeChunk == nullptr)
            {
                return nullptr;
            }
            ProbeChunks.push_back(probeChunk);
            CurrentProbeChunkIndex++;
        }

        DataChunk result;
        if (CurrentMode == ExecutionMode::AVX2)
        {
            result = Avx2Probe(probeChunk);
        }
        else
        {
            result = ScalarProbe(probeChunk);
        }

        if (result != nullptr && result->num_rows() > 0)
        {
            return result;
        }
    }
}

void CuckooHashJoinOperator::BuildHashTable()
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

    // Allocate cuckoo hash table
    HT.Allocate(static_cast<size_t>(TotalBuildRows));

    // Build phase: insert all keys
    int64_t globalRowIdx = 0;

    for (const auto& buildChunk : BuildChunks)
    {
        auto keyColumn = std::static_pointer_cast<arrow::Int32Array>(
            buildChunk->column(BuildKeyColumn));

        const int32_t* keys = keyColumn->raw_values();
        int64_t numRows = buildChunk->num_rows();

        for (int64_t i = 0; i < numRows; i++)
        {
            int32_t key = keys[i];
            if (!ScalarInsert(key, static_cast<int32_t>(globalRowIdx + i)))
            {
                // Insertion failed (cycle detected) - this is rare with good load factor
                LOG_WARNING("Cuckoo hash insertion failed - cycle detected. Consider increasing table size.");
            }
        }

        globalRowIdx += numRows;
    }
}

bool CuckooHashJoinOperator::ScalarInsert(int32_t key, int32_t rowIndex)
{
    // Try to insert at position from hash1
    size_t pos1 = HT.Hash1(key);
    if (HT.Keys[pos1] == CuckooHashTable::EMPTY_KEY)
    {
        HT.Keys[pos1] = key;
        HT.RowIndices[pos1] = rowIndex;
        return true;
    }

    // Try to insert at position from hash2
    size_t pos2 = HT.Hash2(key);
    if (HT.Keys[pos2] == CuckooHashTable::EMPTY_KEY)
    {
        HT.Keys[pos2] = key;
        HT.RowIndices[pos2] = rowIndex;
        return true;
    }

    // Both positions occupied - need to kick out and relocate
    int32_t currentKey = key;
    int32_t currentRowIdx = rowIndex;
    size_t currentPos = pos1;
    bool useHash1 = true;

    for (int kick = 0; kick < CuckooHashTable::MAX_KICKS; kick++)
    {
        // Swap current key with key at current position
        int32_t evictedKey = HT.Keys[currentPos];
        int32_t evictedRowIdx = HT.RowIndices[currentPos];

        HT.Keys[currentPos] = currentKey;
        HT.RowIndices[currentPos] = currentRowIdx;

        currentKey = evictedKey;
        currentRowIdx = evictedRowIdx;

        // Find alternate position for evicted key
        size_t alt1 = HT.Hash1(currentKey);
        size_t alt2 = HT.Hash2(currentKey);

        // Move to the OTHER position (not where we just evicted from)
        if (currentPos == alt1)
        {
            currentPos = alt2;
        }
        else
        {
            currentPos = alt1;
        }

        // Check if alternate position is empty
        if (HT.Keys[currentPos] == CuckooHashTable::EMPTY_KEY)
        {
            HT.Keys[currentPos] = currentKey;
            HT.RowIndices[currentPos] = currentRowIdx;
            return true;
        }
    }

    // Cycle detected - insertion failed
    return false;
}

std::pair<size_t, int64_t> CuckooHashJoinOperator::GetBuildChunkAndLocalIndex(int64_t globalRowIndex) const
{
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
// SCALAR PROBE
// ============================================================================

DataChunk CuckooHashJoinOperator::ScalarProbe(const DataChunk& probeChunk)
{
    auto probeKeyColumn = std::static_pointer_cast<arrow::Int32Array>(
        probeChunk->column(ProbeKeyColumn));

    const int32_t* probeKeys = probeKeyColumn->raw_values();
    int64_t numProbeRows = probeChunk->num_rows();

    std::vector<int64_t> probeMatches;
    std::vector<int64_t> buildMatches;

    for (int64_t i = 0; i < numProbeRows; i++)
    {
        int32_t probeKey = probeKeys[i];

        // Check position 1
        size_t pos1 = HT.Hash1(probeKey);
        if (HT.Keys[pos1] == probeKey)
        {
            probeMatches.push_back(i);
            buildMatches.push_back(HT.RowIndices[pos1]);
            continue;
        }

        // Check position 2
        size_t pos2 = HT.Hash2(probeKey);
        if (HT.Keys[pos2] == probeKey)
        {
            probeMatches.push_back(i);
            buildMatches.push_back(HT.RowIndices[pos2]);
        }
    }

    if (probeMatches.empty())
    {
        return nullptr;
    }

    // Build output
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

// ============================================================================
// AVX2 SIMD PROBE (Polychroniou et al. approach for cuckoo hashing)
// Key insight: Check BOTH hash positions in parallel using SIMD
// ============================================================================

DataChunk CuckooHashJoinOperator::Avx2Probe(const DataChunk& probeChunk)
{
    auto probeKeyColumn = std::static_pointer_cast<arrow::Int32Array>(
        probeChunk->column(ProbeKeyColumn));

    const int32_t* probeKeys = probeKeyColumn->raw_values();
    int64_t numProbeRows = probeChunk->num_rows();

    std::vector<int64_t> probeMatches;
    std::vector<int64_t> buildMatches;

    // SIMD constants for hash computation
    const __m256i vPrime1 = _mm256_set1_epi32(CuckooHashTable::PRIME1);
    const __m256i vPrime2 = _mm256_set1_epi32(CuckooHashTable::PRIME2);
    const __m256i vMask = _mm256_set1_epi32(static_cast<int32_t>(HT.Mask));
    const __m256i vCapacity = _mm256_set1_epi32(static_cast<int32_t>(HT.Capacity));

    int64_t i = 0;

    // Process 8 probe keys at a time
    for (; i + 8 <= numProbeRows; i += 8)
    {
        // Load 8 probe keys
        __m256i vProbeKeys = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&probeKeys[i]));

        // Compute hash1 positions: (key * PRIME1) & mask
        __m256i vHash1 = _mm256_mullo_epi32(vProbeKeys, vPrime1);
        __m256i vPos1 = _mm256_and_si256(vHash1, vMask);

        // Compute hash2 positions: capacity + ((key * PRIME2) & mask)
        __m256i vHash2 = _mm256_mullo_epi32(vProbeKeys, vPrime2);
        __m256i vPos2 = _mm256_add_epi32(vCapacity, _mm256_and_si256(vHash2, vMask));

        // Gather keys from both positions
        __m256i vKeys1 = _mm256_i32gather_epi32(HT.Keys.data(), vPos1, 4);
        __m256i vKeys2 = _mm256_i32gather_epi32(HT.Keys.data(), vPos2, 4);

        // Compare probe keys with gathered keys
        __m256i vMatch1 = _mm256_cmpeq_epi32(vProbeKeys, vKeys1);
        __m256i vMatch2 = _mm256_cmpeq_epi32(vProbeKeys, vKeys2);

        // Extract match masks
        uint32_t matchMask1 = static_cast<uint32_t>(_mm256_movemask_ps(
            _mm256_castsi256_ps(vMatch1)));
        uint32_t matchMask2 = static_cast<uint32_t>(_mm256_movemask_ps(
            _mm256_castsi256_ps(vMatch2)));

        // Extract positions for gathering row indices
        alignas(32) int32_t positions1[8];
        alignas(32) int32_t positions2[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(positions1), vPos1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(positions2), vPos2);

        // Process matches from position 1
        uint32_t mask1 = matchMask1;
        while (mask1)
        {
            int lane = __builtin_ctz(mask1);
            probeMatches.push_back(i + lane);
            buildMatches.push_back(HT.RowIndices[positions1[lane]]);
            mask1 &= ~(1u << lane);
        }

        // Process matches from position 2 (only if not already matched in position 1)
        uint32_t mask2 = matchMask2 & ~matchMask1;  // Exclude already matched
        while (mask2)
        {
            int lane = __builtin_ctz(mask2);
            probeMatches.push_back(i + lane);
            buildMatches.push_back(HT.RowIndices[positions2[lane]]);
            mask2 &= ~(1u << lane);
        }
    }

    // Handle remaining rows with scalar code
    for (; i < numProbeRows; i++)
    {
        int32_t probeKey = probeKeys[i];

        size_t pos1 = HT.Hash1(probeKey);
        if (HT.Keys[pos1] == probeKey)
        {
            probeMatches.push_back(i);
            buildMatches.push_back(HT.RowIndices[pos1]);
            continue;
        }

        size_t pos2 = HT.Hash2(probeKey);
        if (HT.Keys[pos2] == probeKey)
        {
            probeMatches.push_back(i);
            buildMatches.push_back(HT.RowIndices[pos2]);
        }
    }

    if (probeMatches.empty())
    {
        return nullptr;
    }

    // Build output (same as scalar)
    std::vector<std::shared_ptr<arrow::Array>> outputArrays;

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
