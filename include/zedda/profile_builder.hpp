#pragma once

#include <string>
#include <vector>
#include <chrono>
#include "zedda/profile_result.hpp"
#include "zedda/column_accumulator.hpp"
#include "zedda/stream_reader.hpp"
#include "zedda/hyperloglog.hpp"

namespace zedda {

// ─────────────────────────────────────────────────────────────────
//  ProfileBuilder
//
//  One-stop entry point:
//    ProfileBuilder builder("data.csv");
//    DatasetProfile result = builder.build();
//
//  Internally:
//  1. Opens file via CsvStreamReader
//  2. Streams chunks, updating ColumnAccumulators + HyperLogLogs
//  3. Finalizes all accumulators
//  4. Assembles DatasetProfile
//
//  Progress callback (optional):
//    builder.set_progress([](int64_t rows) {
//        std::cout << "\rScanned " << rows << " rows...";
//    });
// ─────────────────────────────────────────────────────────────────
class ProfileBuilder {
public:
    using ProgressCallback = std::function<void(int64_t rows_done)>;

    explicit ProfileBuilder(const std::string&   path,
                            StreamReaderConfig    config = {});

    // Optional: called every chunk with rows scanned so far
    void set_progress(ProgressCallback cb) { progress_cb_ = cb; }

    // Main entry    // Start parallel profiling, returns the final DatasetProfile.
    // If is_sampled is true, each thread will stop early after processing sample_size / num_threads rows.
    DatasetProfile build(bool is_sampled = false, int64_t sample_size = 1000000);

private:
    std::string         path_;
    StreamReaderConfig  config_;
    ProgressCallback    progress_cb_;

    // Build one ColumnProfile from a finalized ColumnAccumulator
    // + its HyperLogLog (for cardinality)
    ColumnProfile make_column_profile(const ColumnAccumulator& acc,
                                      const HyperLogLog&        hll,
                                      int64_t                   total_rows);

    // Compute Pearson correlation matrix over numeric columns
    // Uses running sums — single pass
    void compute_correlations(DatasetProfile& profile);
};

} // namespace zedda