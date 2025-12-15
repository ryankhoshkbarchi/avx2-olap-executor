#include "pch.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
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

    DataChunk Chunk;
    while ((Chunk = Scan.Next()) != nullptr)
    {
        Chunks.push_back(Chunk);
    }
    LOG_MESSAGEF("Loaded %zu chunks into memory.", Chunks.size());
    return Chunks;
}

void PrintUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " <parquet_file> [csv_output] [num_runs]" << std::endl;
    std::cout << "  parquet_file: Path to the Parquet file to benchmark" << std::endl;
    std::cout << "  csv_output:   Path to output CSV file (default: benchmark_uniform_results.csv)" << std::endl;
    std::cout << "  num_runs:     Number of benchmark iterations (default: 10)" << std::endl;
    std::cout << std::endl;
    std::cout << "This benchmark is designed for UNIFORM/NORMALLY distributed data" << std::endl;
    std::cout << "(e.g., NYC TLC taxi trip data with numeric fields)." << std::endl;
    std::cout << std::endl;
    std::cout << "For HashJoin benchmarks, expects companion files:" << std::endl;
    std::cout << "  <base>_build.parquet and <base>_probe.parquet" << std::endl;
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif

    Logger::Init();

    // Parse command line arguments
    std::string TestFile = "nyc_taxi_data.parquet";
    std::string CsvOutput = "benchmark_uniform_results.csv";
    int NumRuns = 10;

    if (argc >= 2)
    {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help")
        {
            PrintUsage(argv[0]);
            return 0;
        }
        TestFile = arg1;
    }
    if (argc >= 3)
    {
        CsvOutput = argv[2];
    }
    if (argc >= 4)
    {
        NumRuns = std::stoi(argv[3]);
    }

    // Check if file exists
    std::ifstream fileCheck(TestFile);
    if (!fileCheck.good())
    {
        std::cerr << "Error: File not found: " << TestFile << std::endl;
        std::cerr << "Run 'python scripts/fetch_nyc_tlc_data.py' to download the data." << std::endl;
        return 1;
    }
    fileCheck.close();

    BenchmarkRunner Runner(NumRuns);
    CSVExporter Exporter(CsvOutput);

    try
    {
        std::vector<DataChunk> InMemoryData = PreloadData(TestFile);
        long long TotalInputRows = 0;
        for (const auto& Chunk : InMemoryData)
        {
            TotalInputRows += Chunk->num_rows();
        }

        LOG_MESSAGEF("Total Input Rows: %lld", TotalInputRows);
        LOG_MESSAGEF("Dataset: %s (Uniform Distribution)", TestFile.c_str());
        LOG_MESSAGEF("CSV Output: %s", CsvOutput.c_str());
        LOG_MESSAGEF("Number of runs: %d", NumRuns);

        std::cout << "============================================================" << std::endl;
        std::cout << "BENCHMARK: UNIFORM DATA (NYC TAXI / NORMAL DISTRIBUTION)" << std::endl;
        std::cout << "============================================================" << std::endl;

        // Calculate a reasonable filter threshold based on data range
        // For uniform data, we'll use a threshold that filters ~50% of the data
        int FilterThreshold = 5000;

        // ============================================================
        // FILTER BENCHMARK
        // ============================================================
        std::cout << "\n--- Filter Operation (threshold > " << FilterThreshold << ") ---" << std::endl;

        auto ScalarFilterPlan = [&]() -> std::unique_ptr<Operator>
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            return std::make_unique<FilterOperator>(std::move(Scan), FilterThreshold, ExecutionMode::SCALAR);
        };

        auto AvxFilterPlan = [&]() -> std::unique_ptr<Operator>
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            return std::make_unique<FilterOperator>(std::move(Scan), FilterThreshold, ExecutionMode::AVX2);
        };

        BenchmarkResult ScalarFilterRes = Runner.Run("Scalar Filter", ScalarFilterPlan, TotalInputRows);
        BenchmarkResult AvxFilterRes = Runner.Run("AVX Filter", AvxFilterPlan, TotalInputRows);
        BenchmarkRunner::PrintComparison("Scalar Filter", ScalarFilterRes.Stats, "AVX Filter", AvxFilterRes.Stats);

        Exporter.WriteResult("Uniform", "Filter", "Scalar", ScalarFilterRes);
        Exporter.WriteResult("Uniform", "Filter", "AVX2", AvxFilterRes);

        // ============================================================
        // SUM BENCHMARK
        // ============================================================
        std::cout << "\n--- Sum Aggregation ---" << std::endl;

        auto ScalarSumPlan = [&]() -> std::unique_ptr<Operator>
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            return std::make_unique<SumOperator>(std::move(Scan), ExecutionMode::SCALAR);
        };

        auto AvxSumPlan = [&]() -> std::unique_ptr<Operator>
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            return std::make_unique<SumOperator>(std::move(Scan), ExecutionMode::AVX2);
        };

        BenchmarkResult ScalarSumRes = Runner.Run("Scalar Sum", ScalarSumPlan, TotalInputRows);
        BenchmarkResult AvxSumRes = Runner.Run("AVX Sum", AvxSumPlan, TotalInputRows);
        BenchmarkRunner::PrintComparison("Scalar Sum", ScalarSumRes.Stats, "AVX Sum", AvxSumRes.Stats);
        BenchmarkRunner::Verify(ScalarSumRes.ResultChunks, AvxSumRes.ResultChunks);

        Exporter.WriteResult("Uniform", "Sum", "Scalar", ScalarSumRes);
        Exporter.WriteResult("Uniform", "Sum", "AVX2", AvxSumRes);

        // ============================================================
        // FILTER + SUM PIPELINE BENCHMARK
        // ============================================================
        std::cout << "\n--- Filter + Sum Pipeline ---" << std::endl;

        auto ScalarFilterSumPlan = [&]() -> std::unique_ptr<Operator>
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            auto Filter = std::make_unique<FilterOperator>(std::move(Scan), FilterThreshold, ExecutionMode::SCALAR);
            return std::make_unique<SumOperator>(std::move(Filter), ExecutionMode::SCALAR);
        };

        auto AvxFilterSumPlan = [&]() -> std::unique_ptr<Operator>
        {
            auto Scan = std::make_unique<MemoryScanOperator>(InMemoryData);
            auto Filter = std::make_unique<FilterOperator>(std::move(Scan), FilterThreshold, ExecutionMode::AVX2);
            return std::make_unique<SumOperator>(std::move(Filter), ExecutionMode::AVX2);
        };

        BenchmarkResult ScalarFilterSumRes = Runner.Run("Scalar Filter+Sum", ScalarFilterSumPlan, TotalInputRows);
        BenchmarkResult AvxFilterSumRes = Runner.Run("AVX Filter+Sum", AvxFilterSumPlan, TotalInputRows);
        BenchmarkRunner::PrintComparison("Scalar Filter+Sum", ScalarFilterSumRes.Stats, "AVX Filter+Sum", AvxFilterSumRes.Stats);
        BenchmarkRunner::Verify(ScalarFilterSumRes.ResultChunks, AvxFilterSumRes.ResultChunks);

        Exporter.WriteResult("Uniform", "FilterSum", "Scalar", ScalarFilterSumRes);
        Exporter.WriteResult("Uniform", "FilterSum", "AVX2", AvxFilterSumRes);

        // ============================================================
        // HASH JOIN BENCHMARK
        // ============================================================
        // Derive join table paths from main file path
        std::string BasePath = TestFile;
        size_t dotPos = BasePath.rfind('.');
        if (dotPos != std::string::npos)
        {
            BasePath = BasePath.substr(0, dotPos);
        }
        std::string BuildTableFile = BasePath + "_build.parquet";
        std::string ProbeTableFile = BasePath + "_probe.parquet";

        std::ifstream buildCheck(BuildTableFile);
        std::ifstream probeCheck(ProbeTableFile);

        if (buildCheck.good() && probeCheck.good())
        {
            buildCheck.close();
            probeCheck.close();

            std::cout << "\n============================================================" << std::endl;
            std::cout << "BENCHMARK SUITE: HASH JOIN (UNIFORM KEY DISTRIBUTION)" << std::endl;
            std::cout << "============================================================" << std::endl;

            LOG_TITLE("SETUP", "Pre-loading hash join data into RAM..");

            std::vector<DataChunk> BuildData = PreloadData(BuildTableFile);
            std::vector<DataChunk> ProbeData = PreloadData(ProbeTableFile);

            LOG_MESSAGEF("Build table: %zu chunks", BuildData.size());
            LOG_MESSAGEF("Probe table: %zu chunks", ProbeData.size());

            long long JoinInputRows = 0;
            for (const auto& Chunk : BuildData)
            {
                JoinInputRows += Chunk->num_rows();
            }
            for (const auto& Chunk : ProbeData)
            {
                JoinInputRows += Chunk->num_rows();
            }

            // Linear Probing Hash Join
            std::cout << "\n--- Linear Probing Hash Join ---" << std::endl;

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

            BenchmarkResult ScalarJoinRes = Runner.Run("Scalar HashJoin", ScalarHashJoinPlan, JoinInputRows);
            BenchmarkResult AvxJoinRes = Runner.Run("AVX HashJoin", AvxHashJoinPlan, JoinInputRows);

            BenchmarkRunner::PrintComparison("Scalar HashJoin", ScalarJoinRes.Stats, "AVX HashJoin", AvxJoinRes.Stats);
            BenchmarkRunner::Verify(ScalarJoinRes.ResultChunks, AvxJoinRes.ResultChunks);

            Exporter.WriteResult("Uniform", "HashJoin_Linear", "Scalar", ScalarJoinRes);
            Exporter.WriteResult("Uniform", "HashJoin_Linear", "AVX2", AvxJoinRes);

            // Cuckoo Hash Join
            std::cout << "\n--- Cuckoo Hash Join ---" << std::endl;

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

            BenchmarkResult ScalarCuckooRes = Runner.Run("Scalar CuckooJoin", ScalarCuckooJoinPlan, JoinInputRows);
            BenchmarkResult AvxCuckooRes = Runner.Run("AVX CuckooJoin", AvxCuckooJoinPlan, JoinInputRows);

            BenchmarkRunner::PrintComparison("Scalar CuckooJoin", ScalarCuckooRes.Stats, "AVX CuckooJoin", AvxCuckooRes.Stats);
            BenchmarkRunner::Verify(ScalarCuckooRes.ResultChunks, AvxCuckooRes.ResultChunks);

            Exporter.WriteResult("Uniform", "HashJoin_Cuckoo", "Scalar", ScalarCuckooRes);
            Exporter.WriteResult("Uniform", "HashJoin_Cuckoo", "AVX2", AvxCuckooRes);

            // Compare Linear vs Cuckoo
            std::cout << "\n--- Comparison: Linear Probing vs Cuckoo Hashing ---" << std::endl;
            BenchmarkRunner::PrintComparison("Linear Probe (AVX)", AvxJoinRes.Stats, "Cuckoo Hash (AVX)", AvxCuckooRes.Stats);
        }
        else
        {
            LOG_TITLE("SKIP", "Hash join test files not found.");
            LOG_MESSAGEF("Expected: %s and %s", BuildTableFile.c_str(), ProbeTableFile.c_str());
            LOG_MESSAGE("Run 'python scripts/fetch_nyc_tlc_data.py --generate-join-tables' to generate test data.");
        }

        // Close exporter to write CSV
        Exporter.Close();

        std::cout << "\n============================================================" << std::endl;
        std::cout << "BENCHMARK COMPLETE" << std::endl;
        std::cout << "Results saved to: " << CsvOutput << std::endl;
        std::cout << "============================================================" << std::endl;

    }
    catch (const std::exception& e)
    {
        LOG_TITLE("CRITICAL ERROR", e.what());
        return 1;
    }

    return 0;
}
