#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>

#include "zedda/profile_builder.hpp"
#include "zedda/profile_result.hpp"
#include "zedda/column_accumulator.hpp"
#include "zedda/arrow_profiler.hpp"

namespace nb = nanobind;
using namespace zedda;

NB_MODULE(fasteda_core, m) {
    m.doc() = "zedda C++ core — blazing fast EDA engine";

    // ── ColumnProfile ─────────────────────────────────────────────
    nb::class_<ColumnProfile>(m, "ColumnProfile")
        .def_rw("name",               &ColumnProfile::name)
        .def_rw("type_str",           &ColumnProfile::type_str)
        .def_rw("total_count",        &ColumnProfile::total_count)
        .def_rw("null_count",         &ColumnProfile::null_count)
        .def_rw("non_null_count",     &ColumnProfile::non_null_count)
        .def_rw("null_pct",           &ColumnProfile::null_pct)
        .def_rw("unique_approx",      &ColumnProfile::unique_approx)
        .def_rw("unique_pct",         &ColumnProfile::unique_pct)
        .def_rw("mean",               &ColumnProfile::mean)
        .def_rw("stddev",             &ColumnProfile::stddev)
        .def_rw("variance",           &ColumnProfile::variance)
        .def_rw("skewness",           &ColumnProfile::skewness)
        .def_rw("kurtosis",           &ColumnProfile::kurtosis)
        .def_rw("val_min",            &ColumnProfile::val_min)
        .def_rw("val_max",            &ColumnProfile::val_max)
        .def_rw("range",              &ColumnProfile::range)
        .def_rw("min_str_len",        &ColumnProfile::min_str_len)
        .def_rw("max_str_len",        &ColumnProfile::max_str_len)
        .def_rw("mean_str_len",       &ColumnProfile::mean_str_len)
        .def_rw("has_high_nulls",     &ColumnProfile::has_high_nulls)
        .def_rw("is_constant",        &ColumnProfile::is_constant)
        .def_rw("is_high_cardinality",&ColumnProfile::is_high_cardinality)
        .def("__repr__", [](const ColumnProfile& c) {
            return "<ColumnProfile '" + c.name + "' (" + c.type_str + ")>";
        });

    // ── DatasetProfile ────────────────────────────────────────────
    nb::class_<DatasetProfile>(m, "DatasetProfile")
        .def_rw("file_name",          &DatasetProfile::file_name)
        .def_rw("file_path",          &DatasetProfile::file_path)
        .def_rw("num_rows",           &DatasetProfile::num_rows)
        .def_rw("num_cols",           &DatasetProfile::num_cols)
        .def_rw("num_numeric",        &DatasetProfile::num_numeric)
        .def_rw("num_string",         &DatasetProfile::num_string)
        .def_rw("overall_null_pct",   &DatasetProfile::overall_null_pct)
        .def_rw("total_null_cells",   &DatasetProfile::total_null_cells)
        .def_rw("total_cells",        &DatasetProfile::total_cells)
        .def_rw("scan_time_ms",       &DatasetProfile::scan_time_ms)
        .def_rw("is_sampled",         &DatasetProfile::is_sampled)
        .def_rw("columns",            &DatasetProfile::columns)
        .def("__repr__", [](const DatasetProfile& d) {
            return "<DatasetProfile '" + d.file_name + "' "
                 + std::to_string(d.num_rows) + " rows x "
                 + std::to_string(d.num_cols) + " cols>";
        });

    // ── profile() — main entry point ─────────────────────────────
    m.def("profile",
        [](const std::string& path, bool show_progress, bool is_sampled, int64_t sample_size) {
            ProfileBuilder builder(path);
            if (show_progress) {
                builder.set_progress([](int64_t rows) {
                    // progress shown from Python side
                    (void)rows;
                });
            }
            return builder.build(is_sampled, sample_size);
        },
        nb::arg("path"),
        nb::arg("show_progress") = true,
        nb::arg("is_sampled") = false,
        nb::arg("sample_size") = 1000000,
        "Profile a CSV/Excel/JSON/Parquet file.\n\n"
        "Example::\n\n"
        "    import zedda as zd\n"
        "    p = zd.profile('data.csv')\n"
        "    print(p.num_rows)\n"
    );

    // ── ArrowProfiler ──────────────────────────────────────────────
    nb::class_<ArrowProfiler>(m, "ArrowProfiler")
        .def(nb::init<const std::string&, int64_t>(), nb::arg("file_name"), nb::arg("total_rows"))
        .def("consume_batch", &ArrowProfiler::consume_batch, nb::arg("schema_ptr"), nb::arg("array_ptr"))
        .def("finalize", &ArrowProfiler::finalize);
}