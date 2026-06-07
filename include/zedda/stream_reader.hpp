#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include "zedda/column_accumulator.hpp"

namespace zedda {

// ─────────────────────────────────────────────────────────────────
//  ChunkResult — what we know after processing one chunk
// ─────────────────────────────────────────────────────────────────
struct ChunkResult {
    int64_t rows_processed = 0;
    int64_t total_rows     = 0;
    bool    done           = false;
};

// ─────────────────────────────────────────────────────────────────
//  StreamReaderConfig — tuning knobs
// ─────────────────────────────────────────────────────────────────
struct StreamReaderConfig {
    int64_t     chunk_size  = 65536;   // rows per chunk (64K default)
    char        delimiter   = ',';
    char        quote_char  = '"';
    bool        has_header  = true;
    std::string null_string = "";      // treat this string as null
};

// ─────────────────────────────────────────────────────────────────
//  CsvStreamReader
//
//  Reads a CSV file chunk by chunk — never loads full file into RAM.
//  For each chunk, updates a vector of ColumnAccumulators.
//  Memory usage = O(num_columns), not O(num_rows).
//
//  Usage:
//    CsvStreamReader reader("data.csv", config);
//    auto accumulators = reader.make_accumulators();
//    while (!reader.done()) {
//        reader.read_chunk(accumulators);
//    }
//    for (auto& acc : accumulators) acc.finalize();
// ─────────────────────────────────────────────────────────────────
class CsvStreamReader {
public:
    explicit CsvStreamReader(const std::string& path,
                             StreamReaderConfig  config = {});
    ~CsvStreamReader();

    // Returns false if file could not be opened
    bool open();
    void close();

    // Build one accumulator per column (call after open())
    std::vector<ColumnAccumulator> make_accumulators() const;

    // Process next chunk — updates accumulators in place
    // Returns info about progress
    ChunkResult read_chunk(std::vector<ColumnAccumulator>& accumulators);

    // Getters
    bool                            done()         const { return done_; }
    int64_t                         rows_read()    const { return rows_read_; }
    const std::vector<std::string>& column_names() const { return col_names_; }
    size_t                          num_columns()  const { return col_names_.size(); }
    const std::string&              path()         const { return path_; }

private:
    std::string         path_;
    StreamReaderConfig  config_;
    FILE*               file_     = nullptr;
    bool                done_     = false;
    int64_t             rows_read_= 0;

    std::vector<std::string> col_names_;
    std::vector<ColumnType>  col_types_;   // detected after first chunk

    // ── internal helpers ─────────────────────────────────────────
    bool        read_header();
    bool        parse_line(const std::string&        line,
                           std::vector<std::string>& fields);
    ColumnType  detect_type(const std::string& sample);
    void        update_accumulator(ColumnAccumulator&  acc,
                                   const std::string&  field,
                                   ColumnType          type);
    bool        is_null(const std::string& field) const;

    // Buffer for reading lines
    std::string line_buf_;
};

} // namespace zedda