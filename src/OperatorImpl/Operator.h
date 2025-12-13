#pragma once
#include <memory>
#include <arrow/record_batch.h>

// a "vector" or batch of rows instead of just 1 row
using DataChunk = std::shared_ptr<arrow::RecordBatch>;

enum class ExecutionMode
{
    SCALAR,
    AVX2,
    AVX512
};

// Base class for all operators (unary operators with single child)
class Operator
{
public:

    Operator(ExecutionMode mode = ExecutionMode::SCALAR)
        : CurrentMode(mode)
    {
    }

    virtual ~Operator() = default;

    virtual DataChunk Next() = 0;  // Every operator must implement Next()

protected:
    ExecutionMode CurrentMode;
};

// Base class for binary operators (two input children, e.g., joins)
class BinaryOperator : public Operator
{
public:
    BinaryOperator(std::unique_ptr<Operator> left,
                   std::unique_ptr<Operator> right,
                   ExecutionMode mode = ExecutionMode::SCALAR)
        : Operator(mode),
          LeftChild(std::move(left)),
          RightChild(std::move(right))
    {
    }

    virtual ~BinaryOperator() = default;

protected:
    std::unique_ptr<Operator> LeftChild;   // Build side for joins
    std::unique_ptr<Operator> RightChild;  // Probe side for joins
};