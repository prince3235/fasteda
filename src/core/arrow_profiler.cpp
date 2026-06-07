#include "zedda/arrow_profiler.hpp"
#include <cstring>
#include <stdexcept>

namespace zedda {

ArrowProfiler::ArrowProfiler(const std::string& file_name, int64_t total_rows)
    : file_name_(file_name), total_rows_(total_rows) {}

ArrowProfiler::~ArrowProfiler() {}

bool ArrowProfiler::is_null(const uint8_t* validity_bitmap, int64_t index) {
    if (!validity_bitmap) return false;
    return (validity_bitmap[index / 8] & (1 << (index % 8))) == 0;
}

void ArrowProfiler::initialize_columns(struct ArrowSchema* schema) {
    int64_t num_cols = schema->n_children;
    accs_.resize(num_cols);
    hlls_.resize(num_cols);
    format_strings_.resize(num_cols);
    
    for (int64_t i = 0; i < num_cols; ++i) {
        struct ArrowSchema* child = schema->children[i];
        accs_[i].name = child->name ? child->name : "";
        format_strings_[i] = child->format ? child->format : "";
        
        std::string_view fmt(format_strings_[i]);
        if (fmt == "c" || fmt == "C" || fmt == "s" || fmt == "S" || 
            fmt == "i" || fmt == "I" || fmt == "l" || fmt == "L") {
            accs_[i].type = ColumnType::INTEGER;
        } else if (fmt == "e" || fmt == "f" || fmt == "g") {
            accs_[i].type = ColumnType::FLOAT;
        } else if (fmt == "b") {
            accs_[i].type = ColumnType::BOOLEAN;
        } else if (fmt == "u" || fmt == "U" || fmt == "z" || fmt == "Z") {
            accs_[i].type = ColumnType::STRING;
        } else if (fmt.length() >= 2 && (fmt.substr(0, 2) == "td" || fmt.substr(0, 2) == "tt" || fmt.substr(0, 2) == "ts")) {
            accs_[i].type = ColumnType::DATETIME;
        } else {
            accs_[i].type = ColumnType::UNKNOWN;
        }
    }
    initialized_ = true;
}

void ArrowProfiler::consume_batch(uintptr_t schema_ptr, uintptr_t array_ptr) {
    struct ArrowSchema* schema = reinterpret_cast<struct ArrowSchema*>(schema_ptr);
    struct ArrowArray* array = reinterpret_cast<struct ArrowArray*>(array_ptr);
    
    if (!initialized_) {
        initialize_columns(schema);
    }

    int64_t num_rows = array->length;
    rows_processed_ += num_rows;

    for (int64_t col = 0; col < array->n_children; ++col) {
        struct ArrowArray* child = array->children[col];
        ColumnType type = accs_[col].type;
        std::string_view fmt = format_strings_[col];

        const uint8_t* validity_bitmap = nullptr;
        if (child->n_buffers > 0 && child->buffers != nullptr && child->buffers[0] != nullptr) {
            validity_bitmap = reinterpret_cast<const uint8_t*>(child->buffers[0]);
        }

        if (child->null_count == num_rows) {
            for (int64_t i = 0; i < num_rows; ++i) accs_[col].update_null();
            continue;
        }

        // We only parse types we care about natively, others become UNKNOWN/NULL equivalent
        if (type == ColumnType::INTEGER) {
            if (fmt == "i") { // int32
                const int32_t* data = child->n_buffers > 1 && child->buffers[1] ? reinterpret_cast<const int32_t*>(child->buffers[1]) : nullptr;
                for (int64_t i = 0; i < num_rows; ++i) {
                    if (is_null(validity_bitmap, i + child->offset) || data == nullptr) accs_[col].update_null();
                    else {
                        double val = static_cast<double>(data[i + child->offset]);
                        accs_[col].update(val);
                        hlls_[col].add(val);
                    }
                }
            } else if (fmt == "l") { // int64
                const int64_t* data = child->n_buffers > 1 && child->buffers[1] ? reinterpret_cast<const int64_t*>(child->buffers[1]) : nullptr;
                for (int64_t i = 0; i < num_rows; ++i) {
                    if (is_null(validity_bitmap, i + child->offset) || data == nullptr) accs_[col].update_null();
                    else {
                        double val = static_cast<double>(data[i + child->offset]);
                        accs_[col].update(val);
                        hlls_[col].add(val);
                    }
                }
            } else {
                for (int64_t i = 0; i < num_rows; ++i) accs_[col].update_null();
            }
        } else if (type == ColumnType::FLOAT) {
            if (fmt == "f") { // float32
                const float* data = child->n_buffers > 1 && child->buffers[1] ? reinterpret_cast<const float*>(child->buffers[1]) : nullptr;
                for (int64_t i = 0; i < num_rows; ++i) {
                    if (is_null(validity_bitmap, i + child->offset) || data == nullptr) accs_[col].update_null();
                    else {
                        double val = static_cast<double>(data[i + child->offset]);
                        accs_[col].update(val);
                        hlls_[col].add(val);
                    }
                }
            } else if (fmt == "g") { // float64
                const double* data = child->n_buffers > 1 && child->buffers[1] ? reinterpret_cast<const double*>(child->buffers[1]) : nullptr;
                for (int64_t i = 0; i < num_rows; ++i) {
                    if (is_null(validity_bitmap, i + child->offset) || data == nullptr) accs_[col].update_null();
                    else {
                        double val = data[i + child->offset];
                        accs_[col].update(val);
                        hlls_[col].add(val);
                    }
                }
            } else {
                for (int64_t i = 0; i < num_rows; ++i) accs_[col].update_null();
            }
        } else if (type == ColumnType::STRING && fmt == "u") {
            const int32_t* offsets = child->n_buffers > 1 && child->buffers[1] ? reinterpret_cast<const int32_t*>(child->buffers[1]) : nullptr;
            const char* str_data = child->n_buffers > 2 && child->buffers[2] ? reinterpret_cast<const char*>(child->buffers[2]) : nullptr;
            
            for (int64_t i = 0; i < num_rows; ++i) {
                int64_t adj_i = i + child->offset;
                if (is_null(validity_bitmap, adj_i) || offsets == nullptr) {
                    accs_[col].update_null();
                } else {
                    int32_t start = offsets[adj_i];
                    int32_t end = offsets[adj_i + 1];
                    int32_t len = end - start;
                    if (str_data != nullptr && len > 0) {
                        std::string_view sv(str_data + start, len);
                        accs_[col].update_string_sv(sv);
                        hlls_[col].add(sv);
                    } else {
                        accs_[col].update_string_sv("");
                        hlls_[col].add(std::string_view(""));
                    }
                }
            }
        } else if (type == ColumnType::STRING && fmt == "U") { // large string
            const int64_t* offsets = child->n_buffers > 1 && child->buffers[1] ? reinterpret_cast<const int64_t*>(child->buffers[1]) : nullptr;
            const char* str_data = child->n_buffers > 2 && child->buffers[2] ? reinterpret_cast<const char*>(child->buffers[2]) : nullptr;
            
            for (int64_t i = 0; i < num_rows; ++i) {
                int64_t adj_i = i + child->offset;
                if (is_null(validity_bitmap, adj_i) || offsets == nullptr) {
                    accs_[col].update_null();
                } else {
                    int64_t start = offsets[adj_i];
                    int64_t end = offsets[adj_i + 1];
                    int64_t len = end - start;
                    if (str_data != nullptr && len > 0) {
                        std::string_view sv(str_data + start, len);
                        accs_[col].update_string_sv(sv);
                        hlls_[col].add(sv);
                    } else {
                        accs_[col].update_string_sv("");
                        hlls_[col].add(std::string_view(""));
                    }
                }
            }
        } else {
            // Unhandled types — record as null
            for (int64_t i = 0; i < num_rows; ++i) accs_[col].update_null();
        }
    }
    // NOTE: Do NOT call schema->release() or array->release() here.
    // PyArrow set those release callbacks and owns the memory.
    // Calling release() from C++ would cause a double-free / crash.
    // PyArrow frees the structs when the Python batch object is GC'd.
}

DatasetProfile ArrowProfiler::finalize() {
    DatasetProfile profile;
    profile.file_name = file_name_;
    profile.file_path = file_name_;
    profile.num_rows = total_rows_;
    profile.num_cols = accs_.size();
    profile.scan_time_ms = 0;

    int64_t total_null_cells = 0;
    for (size_t i = 0; i < accs_.size(); ++i) {
        accs_[i].finalize();
        
        ColumnProfile cp;
        cp.name = accs_[i].name;
        cp.type_str = column_type_str(accs_[i].type);
        cp.total_count = accs_[i].count;
        cp.null_count = accs_[i].null_count;
        cp.non_null_count = accs_[i].non_null_count();
        cp.null_pct = accs_[i].null_pct;
        cp.unique_approx = hlls_[i].count();
        cp.unique_pct = (cp.non_null_count > 0) ? (100.0 * cp.unique_approx / cp.non_null_count) : 0.0;
        
        if (accs_[i].type == ColumnType::INTEGER || accs_[i].type == ColumnType::FLOAT || accs_[i].type == ColumnType::BOOLEAN) {
            cp.mean = accs_[i].mean;
            cp.stddev = accs_[i].stddev;
            cp.variance = accs_[i].variance;
            cp.skewness = accs_[i].skewness;
            cp.kurtosis = accs_[i].kurtosis;
            if (cp.non_null_count > 0) {
                cp.val_min = accs_[i].val_min;
                cp.val_max = accs_[i].val_max;
                cp.range = accs_[i].range();
            }
        }
        
        if (accs_[i].type == ColumnType::STRING || accs_[i].type == ColumnType::DATETIME) {
            if (cp.non_null_count > 0) {
                cp.min_str_len = accs_[i].min_str_len;
                cp.max_str_len = accs_[i].max_str_len;
                cp.mean_str_len = accs_[i].mean_str_len;
            }
        }
        
        cp.has_high_nulls = cp.null_pct > 20.0;
        cp.is_constant = cp.unique_approx <= 1;
        cp.is_high_cardinality = cp.unique_pct > 90.0;
        
        total_null_cells += cp.null_count;
        
        if (cp.type_str == "int" || cp.type_str == "float" || cp.type_str == "bool") {
            profile.num_numeric++;
        } else {
            profile.num_string++;
        }
        
        profile.columns.push_back(std::move(cp));
    }
    
    profile.total_cells = profile.num_rows * profile.num_cols;
    profile.total_null_cells = total_null_cells;
    profile.overall_null_pct = (profile.total_cells > 0) ? (100.0 * static_cast<double>(total_null_cells) / profile.total_cells) : 0.0;
    
    return profile;
}

} // namespace zedda
