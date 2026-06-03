#include <iostream>
#include <iomanip>
#include <cmath>
#include "fasteda/column_accumulator.hpp"
#include "fasteda/hyperloglog.hpp"

// ── Test 1: Welford stats on known data ──────────────────────────
void test_welford() {
    std::cout << "\n=== Test: ColumnAccumulator (Welford) ===\n";

    fasteda::ColumnAccumulator acc;
    acc.name = "age";
    acc.type = fasteda::ColumnType::INTEGER;

    // Known dataset: {2, 4, 4, 4, 5, 5, 7, 9}
    // mean=5, variance=4, stddev=2  (textbook example)
    double data[] = {2, 4, 4, 4, 5, 5, 7, 9};
    for (double v : data) acc.update(v);

    acc.finalize();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Count    : " << acc.count           << " (expected 8)\n";
    std::cout << "Mean     : " << acc.mean             << " (expected 5.0)\n";
    std::cout << "Variance : " << acc.variance         << " (expected 4.571...)\n";
    std::cout << "Stddev   : " << acc.stddev           << " (expected 2.138...)\n";
    std::cout << "Min      : " << acc.val_min          << " (expected 2.0)\n";
    std::cout << "Max      : " << acc.val_max          << " (expected 9.0)\n";
    std::cout << "Null pct : " << acc.null_pct         << "% (expected 0.0%)\n";

    // Verify
    bool ok = std::abs(acc.mean - 5.0) < 1e-9
           && std::abs(acc.val_min - 2.0) < 1e-9
           && std::abs(acc.val_max - 9.0) < 1e-9
           && acc.null_count == 0;
    std::cout << (ok ? "PASS ✓" : "FAIL ✗") << "\n";
}

// ── Test 2: Null handling ─────────────────────────────────────────
void test_nulls() {
    std::cout << "\n=== Test: Null handling ===\n";

    fasteda::ColumnAccumulator acc;
    acc.name = "salary";
    acc.type = fasteda::ColumnType::FLOAT;

    acc.update(1000.0);
    acc.update_null();
    acc.update(2000.0);
    acc.update_null();
    acc.update(3000.0);

    acc.finalize();

    std::cout << "Count      : " << acc.count       << " (expected 5)\n";
    std::cout << "Null count : " << acc.null_count   << " (expected 2)\n";
    std::cout << "Null pct   : " << acc.null_pct     << "% (expected 40.0%)\n";
    std::cout << "Mean       : " << acc.mean         << " (expected 2000.0)\n";

    bool ok = acc.count == 5
           && acc.null_count == 2
           && std::abs(acc.null_pct - 40.0) < 1e-6
           && std::abs(acc.mean - 2000.0) < 1e-6;
    std::cout << (ok ? "PASS ✓" : "FAIL ✗") << "\n";
}

// ── Test 3: HyperLogLog cardinality ──────────────────────────────
void test_hll() {
    std::cout << "\n=== Test: HyperLogLog ===\n";

    fasteda::HyperLogLog hll;

    // Add 10000 unique integers
    for (int i = 0; i < 10000; ++i) {
        hll.add(static_cast<int64_t>(i));
    }

    double est = hll.estimate();
    double error_pct = std::abs(est - 10000.0) / 10000.0 * 100.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "True count : 10000\n";
    std::cout << "Estimated  : " << est << "\n";
    std::cout << "Error      : " << error_pct << "% (expected < 2%)\n";

    // HLL guarantee: < 2% error with 16K registers
    bool ok = error_pct < 2.0;
    std::cout << (ok ? "PASS ✓" : "FAIL ✗") << "\n";
}

// ── Test 4: HLL on strings ───────────────────────────────────────
void test_hll_strings() {
    std::cout << "\n=== Test: HyperLogLog strings ===\n";

    fasteda::HyperLogLog hll;

    // 5 unique cities, repeated many times
    std::string cities[] = {"Mumbai", "Delhi", "Bangalore", "Pune", "Ahmedabad"};
    for (int i = 0; i < 100000; ++i) {
        hll.add(cities[i % 5]);
    }

    int64_t est = hll.count();
    std::cout << "True unique : 5\n";
    std::cout << "Estimated   : " << est << "\n";

    bool ok = (est >= 4 && est <= 6);
    std::cout << (ok ? "PASS ✓" : "FAIL ✗") << "\n";
}

int main() {
    std::cout << "fasteda — Day 1 unit tests\n";
    std::cout << "==========================\n";

    test_welford();
    test_nulls();
    test_hll();
    test_hll_strings();

    std::cout << "\nDone! Agar sab PASS hain toh Week 1 shuru ho gayi! 🚀\n";
    return 0;
}