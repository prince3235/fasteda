#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <string_view>

namespace fasteda {

// ─────────────────────────────────────────────────────────────────
//  HyperLogLog — approximate cardinality (unique value count)
//
//  Why not exact count?
//  Exact unique counting needs a hash set — O(n) memory.
//  On a 10M row column with 1M unique strings, that's hundreds
//  of MB just for one column.
//
//  HyperLogLog gives ~98% accuracy using only 16KB per column,
//  regardless of how many unique values exist.
//
//  Algorithm (Flajolet et al. 2007):
//  - Hash each value to a 64-bit integer
//  - Use top `b` bits to pick one of 2^b registers
//  - In remaining bits, count leading zeros + 1
//  - Store max leading-zeros seen per register
//  - Final estimate = harmonic mean formula * correction factor
//
//  We use b=14 → 2^14 = 16384 registers = 16KB memory
//  Error rate = 1.04 / sqrt(16384) ≈ 0.81%
// ─────────────────────────────────────────────────────────────────

class HyperLogLog {
public:
    static constexpr int    BITS      = 14;
    static constexpr int    NUM_REGS  = 1 << BITS;          // 16384
    static constexpr double ALPHA     = 0.7213 / (1.0 + 1.079 / NUM_REGS);

    HyperLogLog() {
        std::memset(registers_, 0, sizeof(registers_));
    }

    // ─────────────────────────────────────────────────────────────
    //  add(value) — update with a double value
    // ─────────────────────────────────────────────────────────────
    void add(double value) {
        uint64_t h = hash_double(value);
        insert_hash(h);
    }

    // ─────────────────────────────────────────────────────────────
    //  add(value) — update with a string value
    // ─────────────────────────────────────────────────────────────
    void add(std::string_view value) {
        uint64_t h = hash_string(value);
        insert_hash(h);
    }

    void add(const std::string& value) {
        add(std::string_view(value));
    }

    void add(int64_t value) {
        uint64_t h = hash_int(value);
        insert_hash(h);
    }

    // ─────────────────────────────────────────────────────────────
    //  estimate() — returns approximate unique count
    //
    //  Three zones:
    //  1. Small range  (<2.5 * NUM_REGS): LinearCounting correction
    //  2. Normal range: standard HLL formula
    //  3. Large range  (>2^32/30): large-range correction
    // ─────────────────────────────────────────────────────────────
    double estimate() const {
        double sum = 0.0;
        int    zero_regs = 0;

        for (int i = 0; i < NUM_REGS; ++i) {
            sum += 1.0 / (1ULL << registers_[i]);
            if (registers_[i] == 0) ++zero_regs;
        }

        double raw = ALPHA * NUM_REGS * NUM_REGS / sum;

        // Small range correction — LinearCounting
        if (raw <= 2.5 * NUM_REGS && zero_regs > 0) {
            return static_cast<double>(NUM_REGS)
                 * std::log(static_cast<double>(NUM_REGS)
                 / static_cast<double>(zero_regs));
        }

        // Large range correction
        constexpr double TWO_32 = 4294967296.0;
        if (raw > TWO_32 / 30.0) {
            return -TWO_32 * std::log(1.0 - raw / TWO_32);
        }

        return raw;
    }

    // Convenience: rounded integer estimate
    int64_t count() const {
        return static_cast<int64_t>(estimate() + 0.5);
    }

    // Reset all registers
    void reset() {
        std::memset(registers_, 0, sizeof(registers_));
    }

private:
    uint8_t registers_[NUM_REGS];

    // ─────────────────────────────────────────────────────────────
    //  insert_hash — core HLL logic
    //  Top BITS bits → register index
    //  Remaining bits → count leading zeros + 1
    // ─────────────────────────────────────────────────────────────
    void insert_hash(uint64_t h) {
        // Top BITS bits select the register
        uint32_t idx = static_cast<uint32_t>(h >> (64 - BITS));

        // Remaining 64-BITS bits: count leading zeros in them
        // Shift away the index bits, then count leading zeros
        uint64_t w   = (h << BITS) | ((1ULL << BITS) - 1);
        uint8_t  rho = static_cast<uint8_t>(leading_zeros64(w) + 1);

        if (rho > registers_[idx]) {
            registers_[idx] = rho;
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Hash functions
    //  Using MurmurHash3 finalizer (fast, good avalanche)
    // ─────────────────────────────────────────────────────────────
    static uint64_t murmurmix64(uint64_t h) {
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }

    static uint64_t hash_double(double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return murmurmix64(bits);
    }

    static uint64_t hash_int(int64_t v) {
        return murmurmix64(static_cast<uint64_t>(v));
    }

    // FNV-1a 64-bit for strings (fast, minimal collisions)
    static uint64_t hash_string(std::string_view s) {
        uint64_t h = 14695981039346656037ULL;
        for (unsigned char c : s) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        return murmurmix64(h);
    }

    // Portable leading-zero count (replaces __builtin_clzll)
    static int leading_zeros64(uint64_t x) {
        if (x == 0) return 64;
#if defined(_MSC_VER)
        unsigned long idx;
        _BitScanReverse64(&idx, x);
        return 63 - static_cast<int>(idx);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_clzll(x);
#else
        int n = 0;
        if (!(x >> 32)) { n += 32; x <<= 32; }
        if (!(x >> 48)) { n += 16; x <<= 16; }
        if (!(x >> 56)) { n +=  8; x <<=  8; }
        if (!(x >> 60)) { n +=  4; x <<=  4; }
        if (!(x >> 62)) { n +=  2; x <<=  2; }
        if (!(x >> 63)) { n +=  1; }
        return n;
#endif
    }
};

} // namespace fasteda