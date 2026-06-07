#include "zedda/stream_reader.hpp"
#include "zedda/hyperloglog.hpp"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <stdexcept>

namespace zedda {

// ─────────────────────────────────────────────────────────────────
//  Constructor / Destructor
// ─────────────────────────────────────────────────────────────────
CsvStreamReader::CsvStreamReader(const std::string& path,
                                 StreamReaderConfig  config)
    : path_(path), config_(config) {}

CsvStreamReader::~CsvStreamReader() {
    close();
}

// ─────────────────────────────────────────────────────────────────
//  open() — open file and read header row
// ─────────────────────────────────────────────────────────────────
bool CsvStreamReader::open() {
    file_ = fopen(path_.c_str(), "r");
    if (!file_) {
        throw std::runtime_error(
            "Cannot open file: '" + path_ + "'\n"
            "Check: file exists, readable permissions, valid path");
    }

    // Validate it looks like text (not binary) by checking for NULL bytes
    char probe[512];
    size_t n = fread(probe, 1, 512, file_);
    rewind(file_);
    for (size_t i = 0; i < n; i++) {
        if (probe[i] == '\0') {
            throw std::runtime_error(
                "File appears to be binary (contains NULL bytes), not CSV: " + path_);
        }
    }

    if (config_.has_header) {
        if (!read_header()) {
            throw std::runtime_error("[zedda] Failed to read header from: " + path_);
        }
    }

    return true;
}

void CsvStreamReader::close() {
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
}

// ─────────────────────────────────────────────────────────────────
//  make_accumulators() — one per column, named from header
// ─────────────────────────────────────────────────────────────────
std::vector<ColumnAccumulator> CsvStreamReader::make_accumulators() const {
    std::vector<ColumnAccumulator> accs;
    accs.resize(col_names_.size());
    for (size_t i = 0; i < col_names_.size(); ++i) {
        accs[i].name = col_names_[i];
        accs[i].type = ColumnType::UNKNOWN;  // detected on first chunk
    }
    return accs;
}

// ─────────────────────────────────────────────────────────────────
//  read_chunk() — core streaming loop
//
//  Reads up to config_.chunk_size rows.
//  For each row, parses fields and updates accumulators.
//  Type detection happens on first non-null value seen per column.
// ─────────────────────────────────────────────────────────────────
ChunkResult CsvStreamReader::read_chunk(std::vector<ColumnAccumulator>& accs) {
    ChunkResult result;

    if (done_ || !file_) {
        result.done = true;
        return result;
    }

    // Ensure col_types_ is sized correctly
    if (col_types_.size() < accs.size()) {
        col_types_.resize(accs.size(), ColumnType::UNKNOWN);
    }

    std::vector<std::string> fields;
    fields.reserve(accs.size());

    char buf[65536];   // line buffer — 64KB per line max
    int64_t chunk_rows = 0;

    while (chunk_rows < config_.chunk_size) {
        if (fgets(buf, sizeof(buf), file_) == nullptr) {
            // EOF or error
            done_ = true;
            break;
        }

        // Remove trailing newline / carriage return
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) {
            buf[--len] = '\0';
        }

        if (len == 0) continue;  // skip blank lines

        line_buf_.assign(buf, len);
        fields.clear();

        if (!parse_line(line_buf_, fields)) continue;

        // Pad or trim fields to match column count
        fields.resize(accs.size(), "");

        for (size_t col = 0; col < accs.size(); ++col) {
            const std::string& field = fields[col];

            if (is_null(field)) {
                accs[col].update_null();
                continue;
            }

            // Auto-detect type on first non-null value
            if (col_types_[col] == ColumnType::UNKNOWN) {
                col_types_[col] = detect_type(field);
                accs[col].type  = col_types_[col];
            }

            update_accumulator(accs[col], field, col_types_[col]);
        }

        ++chunk_rows;
        ++rows_read_;
    }

    result.rows_processed = chunk_rows;
    result.total_rows     = rows_read_;
    result.done           = done_;
    return result;
}

// ─────────────────────────────────────────────────────────────────
//  read_header() — parse first line as column names
// ─────────────────────────────────────────────────────────────────
bool CsvStreamReader::read_header() {
    char buf[65536];
    if (fgets(buf, sizeof(buf), file_) == nullptr) return false;

    size_t len = strlen(buf);
    while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) {
        buf[--len] = '\0';
    }

    line_buf_.assign(buf, len);
    std::vector<std::string> names;
    if (!parse_line(line_buf_, names)) return false;

    // Trim whitespace from column names
    for (auto& name : names) {
        size_t start = name.find_first_not_of(" \t\"");
        size_t end   = name.find_last_not_of(" \t\"");
        if (start != std::string::npos)
            name = name.substr(start, end - start + 1);
    }

    col_names_ = std::move(names);
    return !col_names_.empty();
}

// ─────────────────────────────────────────────────────────────────
//  parse_line() — RFC 4180 CSV parser
//
//  Handles:
//  - Quoted fields (commas inside quotes)
//  - Escaped quotes ("")
//  - Custom delimiter
// ─────────────────────────────────────────────────────────────────
bool CsvStreamReader::parse_line(const std::string&        line,
                                  std::vector<std::string>& fields) {
    fields.clear();
    std::string field;
    field.reserve(64);

    bool in_quotes = false;
    size_t i = 0;
    const size_t n = line.size();

    while (i < n) {
        char c = line[i];

        if (in_quotes) {
            if (c == config_.quote_char) {
                // Peek ahead: "" = escaped quote
                if (i + 1 < n && line[i+1] == config_.quote_char) {
                    field += c;
                    ++i;  // skip second quote
                } else {
                    in_quotes = false;
                }
            } else {
                field += c;
            }
        } else {
            if (c == config_.quote_char) {
                in_quotes = true;
            } else if (c == config_.delimiter) {
                fields.push_back(field);
                field.clear();
            } else {
                field += c;
            }
        }
        ++i;
    }

    fields.push_back(field);  // last field
    return true;
}

// ─────────────────────────────────────────────────────────────────
//  detect_type() — infer ColumnType from a string sample
//
//  Order of priority: INTEGER > FLOAT > BOOLEAN > DATETIME > STRING
// ─────────────────────────────────────────────────────────────────
ColumnType CsvStreamReader::detect_type(const std::string& s) {
    if (s.empty()) return ColumnType::UNKNOWN;

    // Boolean check
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "true" || lower == "false" ||
        lower == "yes"  || lower == "no"    ||
        lower == "1"    || lower == "0") {
        return ColumnType::BOOLEAN;
    }

    // Integer check — optional leading sign, then digits only
    size_t start = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    bool is_int = (start < s.size());
    for (size_t i = start; i < s.size() && is_int; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) is_int = false;
    }
    if (is_int) return ColumnType::INTEGER;

    // Float check — try stod
    try {
        size_t pos;
        std::stod(s, &pos);
        if (pos == s.size()) return ColumnType::FLOAT;
    } catch (...) {}

    // Datetime heuristic — contains '-' or '/' and ':' or 'T'
    bool has_date_sep = (s.find('-') != std::string::npos ||
                         s.find('/') != std::string::npos);
    bool has_time_sep = (s.find(':') != std::string::npos ||
                         s.find('T') != std::string::npos);
    if (has_date_sep && s.size() >= 8) return ColumnType::DATETIME;
    if (has_date_sep && has_time_sep)  return ColumnType::DATETIME;

    return ColumnType::STRING;
}

// ─────────────────────────────────────────────────────────────────
//  update_accumulator() — route to correct update method
// ─────────────────────────────────────────────────────────────────
void CsvStreamReader::update_accumulator(ColumnAccumulator& acc,
                                          const std::string& field,
                                          ColumnType         type) {
    switch (type) {
        case ColumnType::INTEGER:
        case ColumnType::FLOAT:
        case ColumnType::BOOLEAN: {
            try {
                double val = std::stod(field);
                acc.update(val);
            } catch (...) {
                acc.update_null();  // parse failed = treat as null
            }
            break;
        }
        case ColumnType::STRING:
        case ColumnType::DATETIME:
        case ColumnType::UNKNOWN:
        default: {
            acc.update_string(field);
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  is_null() — check if field represents a missing value
// ─────────────────────────────────────────────────────────────────
bool CsvStreamReader::is_null(const std::string& field) const {
    if (field.empty())                    return true;
    if (field == config_.null_string)     return true;
    // Common null representations
    if (field == "NA"   || field == "N/A") return true;
    if (field == "null" || field == "NULL") return true;
    if (field == "nan"  || field == "NaN")  return true;
    if (field == "none" || field == "None") return true;
    if (field == "#N/A" || field == "?")    return true;
    return false;
}

} // namespace zedda