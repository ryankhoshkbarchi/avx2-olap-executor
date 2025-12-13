#include "pch.h"
#include <iostream>
#include <vector>
#include <fstream>
#include "Misc/Logger.h"
#include "OperatorImpl/ScanOperator.h"
#include "OperatorImpl/MemoryScanOperator.h"
#include "OperatorImpl/AggregateFunctions/SumOperator.h"
#include "OperatorImpl/FilterOperator.h"
#include "OperatorImpl/HashJoinOperator.h"
#include "OperatorImpl/CuckooHashJoinOperator.h"
#include "Benchmarking/BenchmarkRunner.h"



// Moves all data from disk into RAM to avoid expensive I/O during benchmarking
std::vector<DataChunk> PreloadData(const std::string& FileName)
{
    LOG_TITLE("SETUP", "Pre-loading data into RAM..");
    std::vector<DataChunk> Chunks;
    ScanOperator Scan(FileName);

    DataChunk Chunk; // batch of rows
    while ((Chunk = Scan.Next()) != nullptr)
    {
        Chunks.push_back(Chunk);
    }
    LOG_MESSAGEF("Loaded %zu chunks into memory.", Chunks.size());
    return Chunks;
}
// TODO:
// - Change benchmarking logic to only measure CPU time
// - Implement more Aggregate functions (AVG, COUNT, MIN, MAX)
int main()
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif

    Logger::Init();

    std::string TestFile = "TEST_DATA.parquet";
    BenchmarkRunner Runner(10);

    try
    {
        std::vector<DataChunk> InMemoryData = PreloadData(TestFile);
        long long TotalInputRows = 0;
        for (const auto& Chunk : InMemoryData)
        {
            TotalInputRows += Chunk->num_rows();
        }

        LOG_MESSAGEF("Total Input Rows: %lld", TotalInputRows);

        std::cout << "============================================================" << std::endl;
        std::cout << "BENCHMARK SUITE: IN-MEMORY (CPU BOUND)" << std::endl;
        std::cout << "============================================================" << std::endl;


        auto ScalarFilterPlan = [&]() -> std::unique_ptr<Operator> 
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            return std::make_unique<FilterOperator>(std::move(Scan), 5000, ExecutionMode::SCALAR);
		};

        auto AvxFilterPlan = [&]() -> std::unique_ptr<Operator> 
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            return std::make_unique<FilterOperator>(std::move(Scan), 5000, ExecutionMode::AVX2);
		};

		BenchmarkResult ScalarFilterRes = Runner.Run("Scalar Filter", ScalarFilterPlan, TotalInputRows);
		BenchmarkResult AvxFilterRes = Runner.Run("AVX Filter", AvxFilterPlan, TotalInputRows);
		BenchmarkRunner::PrintComparison("Scalar Filter", ScalarFilterRes.Stats, "AVX Filter", AvxFilterRes.Stats);


        // Returns a unique pointer to the root of the operator tree
        auto ScalarSumPlan = [&]() -> std::unique_ptr<Operator> 
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            return std::make_unique<SumOperator>(std::move(Scan), ExecutionMode::SCALAR);
        };

        auto AvxSumPlan = [&]() -> std::unique_ptr<Operator> 
        {
            // Define the Source Operator (lowest in the pipeline)
            // The 'MemoryScanOperator' is used here specifically to read the data
            // from the pre-loaded 'InMemoryData' vector (captured by reference '[&]').
            // This isolates the benchmark to measure only the CPU computation time,
            // excluding slow disk I/O and file parsing overhead.
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);

            // Define the Aggregation Operator (the root of this simple pipeline)
            // This creates the SumOperator, which will pull data from the ScanOperator.
            // std::move(Scan) it transfers ownership of the 'Scan' operator
            // to the 'SumOperator', forming the single-path pipeline (Scan -> Sum).
            return std::make_unique<SumOperator>(std::move(Scan), ExecutionMode::AVX2 );
        };

        BenchmarkResult ScalarSumRes = Runner.Run("Scalar Sum", ScalarSumPlan, TotalInputRows);
        BenchmarkResult AvxSumRes = Runner.Run("AVX Sum", AvxSumPlan, TotalInputRows);

        BenchmarkRunner::PrintComparison("Scalar Sum", ScalarSumRes.Stats, "AVX Sum", AvxSumRes.Stats);
        BenchmarkRunner::Verify(ScalarSumRes.ResultChunks, AvxSumRes.ResultChunks);

        // ============================================================
        // HASH JOIN BENCHMARK
        // ============================================================
        std::cout << "\n============================================================" << std::endl;
        std::cout << "BENCHMARK SUITE: HASH JOIN" << std::endl;
        std::cout << "============================================================" << std::endl;

        // Try to load join test data
        std::string BuildTableFile = "build_table.parquet";
        std::string ProbeTableFile = "probe_table.parquet";

        // Check if join test files exist
        std::ifstream buildCheck(BuildTableFile);
        std::ifstream probeCheck(ProbeTableFile);

        if (buildCheck.good() && probeCheck.good())
        {
            buildCheck.close();
            probeCheck.close();

            LOG_TITLE("SETUP", "Pre-loading hash join data into RAM..");

            std::vector<DataChunk> BuildData = PreloadData(BuildTableFile);
            std::vector<DataChunk> ProbeData = PreloadData(ProbeTableFile);

            LOG_MESSAGEF("Build table: %zu chunks", BuildData.size());
            LOG_MESSAGEF("Probe table: %zu chunks", ProbeData.size());
            TotalInputRows = 0;
            for (const auto& Chunk : BuildData)
            {
                TotalInputRows += Chunk->num_rows();
            }
            for (const auto& Chunk : ProbeData)
            {
                TotalInputRows += Chunk->num_rows();
            }

            // Define hash join plans
            // Note: key column is column 0 in both tables
            auto ScalarHashJoinPlan = [&]() -> std::unique_ptr<Operator>
            {
                auto Build = std::make_unique<MemoryScanOperator>(BuildData);
                auto Probe = std::make_unique<MemoryScanOperator>(ProbeData);
                return std::make_unique<HashJoinOperator>(
                    std::move(Build), std::move(Probe),
                    0, 0,  // key columns
                    ExecutionMode::SCALAR);
            };

            auto AvxHashJoinPlan = [&]() -> std::unique_ptr<Operator>
            {
                auto Build = std::make_unique<MemoryScanOperator>(BuildData);
                auto Probe = std::make_unique<MemoryScanOperator>(ProbeData);
                return std::make_unique<HashJoinOperator>(
                    std::move(Build), std::move(Probe),
                    0, 0,  // key columns
                    ExecutionMode::AVX2);
            };

            BenchmarkResult ScalarJoinRes = Runner.Run("Scalar HashJoin", ScalarHashJoinPlan, TotalInputRows);
            BenchmarkResult AvxJoinRes = Runner.Run("AVX HashJoin", AvxHashJoinPlan, TotalInputRows);

            BenchmarkRunner::PrintComparison("Scalar HashJoin", ScalarJoinRes.Stats, "AVX HashJoin", AvxJoinRes.Stats);
            BenchmarkRunner::Verify(ScalarJoinRes.ResultChunks, AvxJoinRes.ResultChunks);

            // ============================================================
            // CUCKOO HASH JOIN BENCHMARK
            // ============================================================
            std::cout << "\n============================================================" << std::endl;
            std::cout << "BENCHMARK SUITE: CUCKOO HASH JOIN" << std::endl;
            std::cout << "============================================================" << std::endl;

            auto ScalarCuckooJoinPlan = [&]() -> std::unique_ptr<Operator>
            {
                auto Build = std::make_unique<MemoryScanOperator>(BuildData);
                auto Probe = std::make_unique<MemoryScanOperator>(ProbeData);
                return std::make_unique<CuckooHashJoinOperator>(
                    std::move(Build), std::move(Probe),
                    0, 0,
                    ExecutionMode::SCALAR);
            };

            auto AvxCuckooJoinPlan = [&]() -> std::unique_ptr<Operator>
            {
                auto Build = std::make_unique<MemoryScanOperator>(BuildData);
                auto Probe = std::make_unique<MemoryScanOperator>(ProbeData);
                return std::make_unique<CuckooHashJoinOperator>(
                    std::move(Build), std::move(Probe),
                    0, 0,
                    ExecutionMode::AVX2);
            };

            BenchmarkResult ScalarCuckooRes = Runner.Run("Scalar CuckooJoin", ScalarCuckooJoinPlan, TotalInputRows);
            BenchmarkResult AvxCuckooRes = Runner.Run("AVX CuckooJoin", AvxCuckooJoinPlan, TotalInputRows);

            BenchmarkRunner::PrintComparison("Scalar CuckooJoin", ScalarCuckooRes.Stats, "AVX CuckooJoin", AvxCuckooRes.Stats);
            BenchmarkRunner::Verify(ScalarCuckooRes.ResultChunks, AvxCuckooRes.ResultChunks);

            // Compare linear probing vs cuckoo hashing
            std::cout << "\n============================================================" << std::endl;
            std::cout << "COMPARISON: Linear Probing vs Cuckoo Hashing" << std::endl;
            std::cout << "============================================================" << std::endl;
            BenchmarkRunner::PrintComparison("Linear Probe (AVX)", AvxJoinRes.Stats, "Cuckoo Hash (AVX)", AvxCuckooRes.Stats);
        }
        else
        {
            LOG_TITLE("SKIP", "Hash join test files not found.");
            LOG_MESSAGE("Run 'python scripts/generate_join_data.py' to generate test data.");
        }

    }
    catch (const std::exception& e)
    {
        LOG_TITLE("CRITICAL ERROR", e.what());
        return 1;
    }

    std::cin.get();
    return 0;
}
