#include "zedda/profile_builder.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace zedda {

ProfileBuilder::ProfileBuilder(const std::string&  path,
                               StreamReaderConfig   config)
    : path_(path), config_(config) {}

// ─────────────────────────────────────────────────────────────────
//  build() — full pipeline
// ─────────────────────────────────────────────────────────────────
DatasetProfile ProfileBuilder::build() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // ── 1. Open reader ───────────────────────────────────────────
    CsvStreamReader reader(path_, config_);
    if (!reader.open()) {
        throw std::runtime_error("[zedda] Cannot open: " + path_);
    }

    size_t ncols = reader.num_columns();

    // ── 2. Init accumulators + HyperLogLogs ──────────────────────
    auto accs = reader.make_accumulators();
    std::vector<HyperLogLog> hlls(ncols);

    // ── 3. Stream all chunks ─────────────────────────────────────
    // We need a second pass for HLL — so we collect raw string
    // values into HLL during the same pass by re-reading each field.
    // Solution: subclass or duplicate? No — we integrate HLL update
    // directly into the chunk loop here.

    // Re-open and re-read manually to integrate HLL:
    reader.close();

    FILE* f = fopen(path_.c_str(), "r");
    if (!f) throw std::runtime_error("[zedda] Cannot open: " + path_);

    char buf[65536];

    // skip header
    if (config_.has_header) {
        fgets(buf, sizeof(buf), f);
    }

    int64_t total_rows = 0;
    std::vector<std::string> fields;
    fields.reserve(ncols);

    // Detect types first chunk approach — we track detected types
    std::vector<ColumnType> col_types(ncols, ColumnType::UNKNOWN);

    while (fgets(buf, sizeof(buf), f)) {
        // strip newline
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'))
            buf[--len] = '\0';
        if (len == 0) continue;

        // parse fields
        fields.clear();
        std::string line(buf, len);
        // Simple split (reuse logic inline)
        std::string field;
        bool in_q = false;
        for (size_t i = 0; i <= line.size(); ++i) {
            char c = (i < line.size()) ? line[i] : config_.delimiter;
            if (in_q) {
                if (c == config_.quote_char) {
                    if (i+1 < line.size() && line[i+1] == config_.quote_char) {
                        field += c; ++i;
                    } else { in_q = false; }
                } else { field += c; }
            } else {
                if (c == config_.quote_char) { in_q = true; }
                else if (c == config_.delimiter) {
                    fields.push_back(field); field.clear();
                } else { field += c; }
            }
        }
        fields.resize(ncols, "");

        for (size_t col = 0; col < ncols; ++col) {
            const std::string& fld = fields[col];

            // null check
            bool is_null = fld.empty()
                || fld == "NA" || fld == "N/A"
                || fld == "null" || fld == "NULL"
                || fld == "nan"  || fld == "NaN"
                || fld == "none" || fld == "None"
                || fld == "#N/A" || fld == "?";

            if (is_null) {
                accs[col].update_null();
                continue;
            }

            // type detect
            if (col_types[col] == ColumnType::UNKNOWN) {
                // try int
                bool is_int = !fld.empty();
                size_t st = (fld[0]=='-'||fld[0]=='+') ? 1 : 0;
                for (size_t k = st; k < fld.size() && is_int; ++k)
                    if (!std::isdigit((unsigned char)fld[k])) is_int = false;
                if (is_int && st < fld.size()) {
                    col_types[col] = ColumnType::INTEGER;
                } else {
                    try {
                        size_t pos;
                        std::stod(fld, &pos);
                        if (pos == fld.size()) col_types[col] = ColumnType::FLOAT;
                        else col_types[col] = ColumnType::STRING;
                    } catch(...) {
                        col_types[col] = ColumnType::STRING;
                    }
                }
                accs[col].type = col_types[col];
            }

            // update accumulator
            if (col_types[col] == ColumnType::INTEGER ||
                col_types[col] == ColumnType::FLOAT) {
                try {
                    accs[col].update(std::stod(fld));
                    hlls[col].add(std::stod(fld));
                } catch(...) { accs[col].update_null(); }
            } else {
                accs[col].update_string(fld);
                hlls[col].add(fld);
            }
        }

        ++total_rows;
        if (progress_cb_ && total_rows % 10000 == 0) {
            progress_cb_(total_rows);
        }
    }
    fclose(f);

    // ── 4. Finalize all accumulators ─────────────────────────────
    for (auto& acc : accs) acc.finalize();

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ── 5. Assemble DatasetProfile ────────────────────────────────
    DatasetProfile profile;
    profile.file_path    = path_;
    profile.num_rows     = total_rows;
    profile.num_cols     = static_cast<int64_t>(ncols);
    profile.scan_time_ms = ms;

    // file_name = last component of path
    size_t slash = path_.find_last_of("/\\");
    profile.file_name = (slash == std::string::npos)
                      ? path_ : path_.substr(slash + 1);

    // column names from reader
    // (re-read header to get names)
    {
        FILE* fh = fopen(path_.c_str(), "r");
        if (fh) {
            if (fgets(buf, sizeof(buf), fh)) {
                size_t l = strlen(buf);
                while (l > 0 && (buf[l-1]=='\n'||buf[l-1]=='\r')) buf[--l]='\0';
                std::string hline(buf, l);
                std::string nm;
                size_t col_idx = 0;
                for (size_t i = 0; i <= hline.size(); ++i) {
                    char c = (i < hline.size()) ? hline[i] : ',';
                    if (c == ',') {
                        // trim quotes
                        size_t s = nm.find_first_not_of(" \t\"");
                        size_t e = nm.find_last_not_of(" \t\"");
                        if (s != std::string::npos && col_idx < accs.size())
                            accs[col_idx].name = nm.substr(s, e-s+1);
                        col_idx++;
                        nm.clear();
                    } else { nm += c; }
                }
            }
            fclose(fh);
        }
    }

    int64_t total_null_cells = 0;
    for (size_t i = 0; i < ncols; ++i) {
        auto cp = make_column_profile(accs[i], hlls[i], total_rows);
        total_null_cells += cp.null_count;

        if (cp.type_str == "int" || cp.type_str == "float")
            ++profile.num_numeric;
        else
            ++profile.num_string;

        profile.columns.push_back(std::move(cp));
    }

    profile.total_cells     = total_rows * static_cast<int64_t>(ncols);
    profile.total_null_cells = total_null_cells;
    profile.overall_null_pct = (profile.total_cells > 0)
        ? 100.0 * total_null_cells / profile.total_cells
        : 0.0;

    return profile;
}

// ─────────────────────────────────────────────────────────────────
//  make_column_profile()
// ─────────────────────────────────────────────────────────────────
ColumnProfile ProfileBuilder::make_column_profile(
        const ColumnAccumulator& acc,
        const HyperLogLog&       hll,
        int64_t                  total_rows) {

    ColumnProfile cp;
    cp.name          = acc.name;
    cp.type_str      = column_type_str(acc.type);
    cp.total_count   = acc.count;
    cp.null_count    = acc.null_count;
    cp.non_null_count= acc.non_null_count();
    cp.null_pct      = acc.null_pct;
    cp.unique_approx = hll.count();
    cp.unique_pct    = (acc.non_null_count() > 0)
        ? 100.0 * cp.unique_approx / acc.non_null_count()
        : 0.0;

    if (acc.type == ColumnType::INTEGER || acc.type == ColumnType::FLOAT) {
        cp.mean     = acc.mean;
        cp.stddev   = acc.stddev;
        cp.variance = acc.variance;
        cp.skewness = acc.skewness;
        cp.kurtosis = acc.kurtosis;
        cp.val_min  = acc.val_min;
        cp.val_max  = acc.val_max;
        cp.range    = acc.range();
    }

    if (acc.type == ColumnType::STRING || acc.type == ColumnType::DATETIME) {
        cp.min_str_len  = acc.min_str_len;
        cp.max_str_len  = acc.max_str_len;
        cp.mean_str_len = acc.mean_str_len;
    }

    // health flags
    cp.has_high_nulls      = cp.null_pct > 20.0;
    cp.is_constant         = cp.unique_approx <= 1;
    cp.is_high_cardinality = cp.unique_pct > 90.0;

    return cp;
}

} // namespace zedda