#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "zedda/column_accumulator.hpp"

namespace zedda {

// ─────────────────────────────────────────────────────────────────
//  ColumnProfile — everything we know about one column
// ─────────────────────────────────────────────────────────────────
struct ColumnProfile {
    // identity
    std::string name;
    std::string type_str;       // "int", "float", "str", etc.

    // counts
    int64_t total_count    = 0;
    int64_t null_count     = 0;
    int64_t non_null_count = 0;
    int64_t unique_approx  = 0; // HyperLogLog estimate
    double  null_pct       = 0.0;
    double  unique_pct     = 0.0;

    // numeric stats (only for int/float cols)
    double mean     = 0.0;
    double stddev   = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    double val_min  = 0.0;
    double val_max  = 0.0;
    double range    = 0.0;

    // string stats (only for str cols)
    int64_t min_str_len  = 0;
    int64_t max_str_len  = 0;
    double  mean_str_len = 0.0;

    // health flags (for report highlighting)
    bool has_high_nulls    = false;  // null_pct > 20%
    bool is_constant       = false;  // unique_approx == 1
    bool is_high_cardinality = false; // unique_pct > 90%
};

// ─────────────────────────────────────────────────────────────────
//  DatasetProfile — the complete EDA result
// ─────────────────────────────────────────────────────────────────
struct DatasetProfile {
    // file info
    std::string file_path;
    std::string file_name;

    // shape
    int64_t num_rows    = 0;
    int64_t num_cols    = 0;
    int64_t num_numeric = 0;
    int64_t num_string  = 0;

    // overall health
    double overall_null_pct      = 0.0;
    int64_t total_null_cells     = 0;
    int64_t total_cells          = 0;

    // timing
    double scan_time_ms = 0.0;

    // per-column profiles
    std::vector<ColumnProfile> columns;

    // correlation matrix (numeric cols only)
    // stored as flat vector: correlations[i * num_numeric + j]
    std::vector<double>      correlation_matrix;
    std::vector<std::string> correlation_col_names;
};

} // namespace zedda