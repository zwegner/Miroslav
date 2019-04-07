#pragma once

#include <assert.h>
#include <stdint.h>

#include <iostream>
#include <tuple>
#include <vector>

#include <immintrin.h>

#define UNUSED __attribute__((unused))

#define EXPECT(x, v)     __builtin_expect((x), (v))

static inline uint32_t bsf64(uint64_t x) {
	return __builtin_ctzll(x);
}

static inline uint32_t bsf32(uint32_t x) {
	return __builtin_ctzl(x);
}

#define START_STATE (0)
#define END_STATE (1)

typedef std::vector<std::tuple<uint8_t, uint32_t, uint32_t>> NFAEdgeList;

// Wacky macro to make tuple unpacking a little less annoying
#define FOR_EACH_EDGE(c, from, to, edges) \
    for (auto edge_i = edges.begin(); \
            edge_i != edges.end() ? (std::tie(c, from, to) = *edge_i), 1 : 0; \
            edge_i++)

////////////////////////////////////////////////////////////////////////////////
// Basic mmap wrapper for file handling
////////////////////////////////////////////////////////////////////////////////
struct File {
    const char *path;
    const uint8_t *data;
    size_t size;

    File() : path(NULL), data(NULL), size(0) { }

    File(const char *path) : path(path) {
        FILE *f = fopen(path, "r");

        fseek(f, 0, SEEK_END);
        this->size = ftell(f);

        this->data = (uint8_t *)mmap(NULL, this->size, PROT_READ, MAP_FILE|MAP_PRIVATE, fileno(f), 0);
        if (this->data == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }

        fclose(f);
    }

    ~File() {
        if (munmap((void *)this->data, this->size) == -1) {
            perror("munmap");
            exit(1);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// Vector definition classes
// These contain the basic properties of the underlying SIMD implementation,
// as well as functions for any architecture-dependent operations.
////////////////////////////////////////////////////////////////////////////////

// AVX2: we have 32-byte vectors, but can only shuffle within 16-byte halves of
// these vectors, which requires special handling
struct VecInfoAVX2 {
    static const uint32_t VL = 32;
    typedef __m256i V;
    typedef uint32_t vmask;
    typedef uint64_t double_vmask;

    // Lookup mask: since avx2 shuffles only work in two 16-byte chunks, we have
    // to use just four bits of each character as a table index, instead of the
    // five that we'd prefer
    static const uint32_t LMASK = 16 - 1;

    static inline V permute(V &table, V &index) {
        return _mm256_shuffle_epi8(table, index);
    }

    static inline V vec_and(V &a, V &b) {
        return _mm256_and_si256(a, b);
    }

    static inline V vec_shr(V &a, uint32_t shift) {
        return _mm256_srli_epi32(a, shift);
    }

    static inline V vec_lanes_shl_1(V &top, V &bottom) {
        // Move all the vector lanes in "top" to the left by one and fill
        // in the first lane with the last lane in "bottom". Since AVX2
        // generally works on two separate 16-byte vectors glued together,
        // this needs two steps. The permute takes [bottom_H, bottom_L]
        // and [top_H, top_L] and gives us [top_L, bottom_H]. The align then
        // takes [top_H, top_L] and gives us [top_H[1:], top_L[:1]], and
        // takes [top_L, bottom_H] and gives us [top_L[1:], bottom_H[:1]].
        V shl_16 = _mm256_permute2x128_si256(top, bottom, 0x03);
        return _mm256_alignr_epi8(top, shl_16, 15);
    }

    static inline vmask test_high_bit(V &a) {
        return _mm256_movemask_epi8(a);
    }

    static inline vmask test_low_bit(V &a) {
        // Movemask tests the high bit, so the input has to be shifted up
        return _mm256_movemask_epi8(_mm256_slli_epi32(a, 7));
    }

    static inline vmask test_nz(V &a) {
        return ~_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, _mm256_setzero_si256()));
    }

    static void prepare_state_table(uint8_t state_bytes[VL]) {
        // HACK because AVX2 sucks and can only do 16-byte shuffles
        for (uint32_t i = 0; i < 16; i++)
            state_bytes[i + 16] = state_bytes[i];
    }
};

// Functions specialized on both vector size and element size. C++ doesn't
// allow explicit specializations inside classes, so they're out here...

// Broadcast
template<typename VI, typename element>
inline typename VI::V broadcast(element value);
template<>
inline VecInfoAVX2::V broadcast<VecInfoAVX2, uint8_t>(uint8_t value) {
    return _mm256_set1_epi8(value);
}
template<>
inline VecInfoAVX2::V broadcast<VecInfoAVX2, uint16_t>(uint16_t value) {
    return _mm256_set1_epi16(value);
}
template<>
inline VecInfoAVX2::V broadcast<VecInfoAVX2, uint32_t>(uint32_t value) {
    return _mm256_set1_epi32(value);
}

// Test equal
template<typename VI, typename element>
inline typename VI::vmask test_eq(typename VI::V &a, typename VI::V &b);
template<>
inline VecInfoAVX2::vmask test_eq<VecInfoAVX2, uint8_t>(
        VecInfoAVX2::V &a, VecInfoAVX2::V &b) {
    return _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b));
}
template<>
inline VecInfoAVX2::vmask test_eq<VecInfoAVX2, uint16_t>(
        VecInfoAVX2::V &a, VecInfoAVX2::V &b) {
    // HACK: avx2 doesn't have a movemask_epi16 instruction. So we just use the
    // epi8 version, and in the one place test_eq is used now, we divide the
    // bitscan of this mask by 2.
    return _mm256_movemask_epi8(_mm256_cmpeq_epi16(a, b));
}
template<>
inline VecInfoAVX2::vmask test_eq<VecInfoAVX2, uint32_t>(
        VecInfoAVX2::V &a, VecInfoAVX2::V &b) {
    return _mm256_movemask_ps(_mm256_cmpeq_epi32(a, b));
}

// Lossy bitset. This works pretty much like a Bloom filter. For quickly
// testing membership of a single index within the bitset, we look at
// contiguous 6-bit chunks of the index value, and set a single bit within
// the bitset based on that. For example, given the index 0xc68 (binary
// 110001101000) and shifts of 0, 3, and 6, we get the following values:
//
//  1 1 0 0 0 1[1 0 1 0 0 0]  --> 101000 (40)
//  1 1 0[0 0 1 1 0 1]0 0 0   --> 001101 (13)
// [1 1 0 0 0 1]1 0 1 0 0 0   --> 110001 (49)
//
// So we test bit 40 in the first bitset, 13 in the second, and 49 in the
// third. Only if all of these bits are set do we return true from
// might_contain.
//
// Also note that this implementation depends on 64-bit variable shifts
// being implicitly modulo 64, as they are on Intel chips...
template<const uint32_t N_SHIFTS, const uint32_t SHIFTS[]>
struct _LossyBitset {
    uint64_t bitsets[N_SHIFTS];
    _LossyBitset() : bitsets{0} {
    }
    void add(uint32_t index) {
        for (uint32_t s = 0; s < N_SHIFTS; s++) {
            uint32_t sub_index = index >> SHIFTS[s];
            bitsets[s] |= 1 << sub_index;
        }
    }
    bool might_contain(uint32_t index) {
        for (uint32_t s = 0; s < N_SHIFTS; s++) {
            uint32_t sub_index = index >> SHIFTS[s];
            if (!(bitsets[s] & 1 << sub_index))
                return false;
        }
        return true;
    }
};

static constexpr uint32_t LBS_SHIFTS[] = {0, 4};
static constexpr uint32_t N_LBS_SHIFTS = sizeof(LBS_SHIFTS) / sizeof(LBS_SHIFTS[0]);
typedef _LossyBitset<N_LBS_SHIFTS, LBS_SHIFTS> LossyBitset;

////////////////////////////////////////////////////////////////////////////////
// Match verifier
// Since the vectorized matcher can give false positives, we have to run through
// each potential match backwards from the end character.
////////////////////////////////////////////////////////////////////////////////

// Always return true. Bad for correctness, good for testing speed of the core
// Miroslav algorithm.
struct FakeMatchVerifier {
    FakeMatchVerifier(UNUSED NFAEdgeList &edges) { }

    const uint8_t *verify(UNUSED const uint8_t *data, UNUSED const uint8_t *end) {
        return end;
    }
};

// Basic bitset NFA simulator
template<typename StateInfo>
class BasicMatchVerifier {
    typedef typename StateInfo::smask smask;
    static const uint32_t MAX_STATES = StateInfo::MAX_STATES;

    // This is a table of [input_byte][state] -> prev_states
    smask back_edges[256][MAX_STATES];
    // All states that lead to a successful match from each input byte
    smask match_mask[256];
    // Shortcut mask: for each input byte, keep a mask of states that have a
    // predecessor state so we don't try any unnecessary lookups
    smask next_mask[256];

public:
    BasicMatchVerifier(NFAEdgeList &edges) {
        for (uint32_t i = 0; i < 256; i++) {
            for (uint32_t j = 0; j < MAX_STATES; j++)
                back_edges[i][j] = 0;
            next_mask[i] = 0;
            match_mask[i] = 0;
        }

        // Initialize state mask tables
        uint8_t c;
        uint32_t from, to;
        FOR_EACH_EDGE(c, from, to, edges) {
            assert(from < MAX_STATES);
            assert(to < MAX_STATES);
            next_mask[c] |= (smask)1 << to;

            if (from == START_STATE)
                match_mask[c] |= (smask)1 << to;
            else
                back_edges[c][to] |= (smask)1 << from;
        }
    }

    const uint8_t *verify(const uint8_t *data, const uint8_t *end) {
        smask states = 1 << END_STATE;

        do {
            uint8_t c = *end;

            if (match_mask[c] & states)
                return end;

            states &= next_mask[c];

            // Iterate through all current states and look up their next states
            smask next_states = 0;
            while (states) {
                uint32_t s = StateInfo::bsf(states);
                next_states |= back_edges[c][s];
                states &= states - 1;
            }
            states = next_states;
        } while (states && --end >= data);

        return NULL;
    }
};

// Structs with definitions for the BasicMatchVerifier.
// smask should have MAX_STATES bits.
// The backwards state table will be (max/8) * max * 256 bytes.
// So 8 states is 2K, 16 => 8K, 32 => 32K, 64 => 128K, 128 => 512K

struct StateInfo32 {
    static const uint32_t MAX_STATES = 32;
    typedef uint32_t smask;
    static inline uint32_t bsf(smask x) { return bsf32(x); }
};
struct StateInfo64 {
    static const uint32_t MAX_STATES = 64;
    typedef uint64_t smask;
    static inline uint32_t bsf(smask x) { return bsf64(x); }
};
struct StateInfo128 {
    static const uint32_t MAX_STATES = 128;
    typedef __uint128_t smask;
    static inline uint32_t bsf(smask x) {
        if ((uint64_t)x)
            return bsf64(((uint64_t)x));
        return bsf64(x >> 64) + 64;
    }
};

typedef BasicMatchVerifier<StateInfo32> BasicMatchVerifier32;
typedef BasicMatchVerifier<StateInfo64> BasicMatchVerifier64;
typedef BasicMatchVerifier<StateInfo128> BasicMatchVerifier128;

// Find the next occurrence of 'chr' in the data stream before 'bound', and return
// the pointer of the character after that. We can go in both directions.
inline const uint8_t *skip_chr(const uint8_t *p, const uint8_t chr, bool forwards,
        const uint8_t *bound) {
    if (forwards)
        while (p < bound && *p != chr)
            p++;
    else
        while (p >= bound && *p != chr)
            p--;
    return p;
}

// Dumb empty class. Using Squamatus as a verifier doesn't need a handler, so we
// use this type as a placeholder
struct DummyMatchHandler {
    typedef uint32_t return_type;
    void handle_match(UNUSED File &f, UNUSED const uint8_t *start,
            UNUSED const uint8_t *end) {
    }
};

struct MatcherRegexOpts {
    static const bool FORWARDS = true;
    static const bool CONTINUOUS = true;
    static const bool OVERLAPPING = false;
    static const bool ONE_PER_LINE = false;
};

struct VerifierRegexOpts {
    static const bool FORWARDS = false;
    static const bool CONTINUOUS = false;
    static const bool OVERLAPPING = false;
    static const bool ONE_PER_LINE = false;
};

template<typename VI, typename VE, typename Opts, typename MatchHandler>
class _Squamatus {
    // Hacky substitute for "using"
    typedef typename VI::V V;
    typedef typename VI::vmask vmask;
    static const uint32_t VL = VI::VL;

    // Vector constants
    static const uint64_t N_VE = VI::VL / sizeof(VE);
    static const uint64_t MAX_STATES = ((uint64_t)1 << 8 * sizeof(VE));
    static const uint64_t SENTINEL = MAX_STATES - 1;
    // HACK: we divide by 2 when using 16 bit compares since there's no movemask_epi16
    static const uint64_t MOVEMASK_HACK = sizeof(VE) == 2 ? 2 : 1;
    // Another HACKish thing: for a masking operation deep in the algorithm, we
    // need to mask out all the bits above the ones that might be set by a vcmpeq
    // instruction. This should be all ones (thus a no-op) unless sizeof(VE) > 2.
    static const vmask VCMP_BITS = ~(-1 << N_VE * MOVEMASK_HACK);

    // Associative array of state -> state transitions, with a different array per
    // possible input byte
    uint32_t key_count[256];
    VE *edge_keys[256];
    VE *edge_values[256];

    // For keeping track of all current states and all next states
    VE *_state_buffer[2];

    // For keeping track of where a given NFA match sequence started
    uint32_t *_start_buffer[2];

    // Stuff to act like a full regex matcher
    MatchHandler &match_handler;
    File *input_file;

    static inline uint32_t INITIAL_STATE() {
        return Opts::FORWARDS ? START_STATE : END_STATE;
    }
    static inline uint32_t TARGET_STATE() {
        return Opts::FORWARDS ? END_STATE : START_STATE;
    }

public:
    // Constructor wrapper using a cool NULL reference, for when using this class
    // as a verifier (the handler isn't touched)
    _Squamatus(NFAEdgeList &edges) : _Squamatus(edges, *(MatchHandler *)NULL) {
        assert(!Opts::CONTINUOUS);
    }

    _Squamatus(NFAEdgeList &edges, MatchHandler &handler) : match_handler(handler) {
        // Group NFA edges into vectors by character
        struct kv_pair {
            VE k, v;
            kv_pair(VE k, VE v) : k(k), v(v) { }
        };
        std::vector<struct kv_pair> edge_pairs[256];
        uint8_t c;
        uint32_t from, to;
        FOR_EACH_EDGE(c, from, to, edges) {
            assert((uint64_t)from < SENTINEL);
            assert((uint64_t)to < SENTINEL);
            assert(from != END_STATE);
            assert(to != START_STATE);
            // Add the from/to states into the associative array. Which is
            // the key and which is the value depends on which direction
            // we're going.
            if (Opts::FORWARDS)
                edge_pairs[c].push_back(kv_pair(from, to));
            else
                edge_pairs[c].push_back(kv_pair(to, from));
        }

        uint32_t max_concurrent_states = 0;

        for (uint32_t i = 0; i < 256; i++) {
            // Sort the vector so all states that a given state/character combination
            // can lead to are all contiguous
            std::stable_sort(edge_pairs[i].begin(), edge_pairs[i].end(),
                    [](const auto& a, const auto& b) {
                        return a.k < b.k || (a.k == b.k && a.v < b.v);
                    });

            if (edge_pairs[i].size() == 0) {
                edge_keys[i] = edge_values[i] = NULL;
                key_count[i] = 0;
                continue;
            }

            // Calculate number of elements. We add N_VE and clear the low bits
            // (x & -N_VE rounds x down to the nearest multiple of N_VE) to get
            // the number of bytes we will store the keys/values in. We add
            // N_VE to round up, but also multiples of N_VE round up to the
            // next multiple, so we always get at least one extra slot. That
            // way we can just compare states in a loop while ignoring the
            // array length (since we're storing 255s at the end of the table
            // that will always compare false).
            key_count[i] = (edge_pairs[i].size() + N_VE) & -N_VE;
            edge_keys[i] = (VE *)malloc(key_count[i] * sizeof(VE));
            edge_values[i] = (VE *)malloc(key_count[i] * sizeof(VE));

            // Copy the sorted vector into the key/value lists, filling in the
            // rest of the values with SENTINEL
            uint32_t j;
            for (j = 0; j < edge_pairs[i].size(); j++) {
                edge_keys[i][j] = edge_pairs[i][j].k;
                edge_values[i][j] = edge_pairs[i][j].v;
            }
            for (; j < key_count[i]; j++) {
                edge_keys[i][j] = SENTINEL;
                edge_values[i][j] = SENTINEL;
            }

            // Update the # of max concurrent states. Since we fill in the state
            // buffer only with values from reading the edge_values[] array for
            // a single input character, the maximum length of one of these arrays
            // is an upper bound for the number of states that this NFA can
            // possibly be in at once.
            if (key_count[i] * N_VE > max_concurrent_states)
                max_concurrent_states = key_count[i] * N_VE;
        }

        // Allocate two buffers for storing the current states
        // +VL because we can write past the end of this array
        size_t buf_size = max_concurrent_states * sizeof(VE) + VL;
        _state_buffer[0] = (VE *)malloc(buf_size);
        _state_buffer[1] = (VE *)malloc(buf_size);

        // Allocate buffers for storing the start index of each
        // match as well. We have to allocate a bit more scratch
        // space at the end, since with this array we could potentially
        // write up to 4*VL bytes off the end
        size_t start_buf_size = max_concurrent_states * sizeof(uint32_t);
        start_buf_size += (VL * sizeof(uint32_t) / sizeof(VE));
        _start_buffer[0] = (uint32_t *)malloc(start_buf_size);
        _start_buffer[1] = (uint32_t *)malloc(start_buf_size);
    }

    ~_Squamatus() {
        for (uint32_t i = 0; i < 256; i++) {
            if (key_count[i]) {
                free(edge_keys[i]);
                free(edge_values[i]);
            }
        }
        for (uint32_t i = 0; i < 2; i++) {
            free(_state_buffer[i]);
            free(_start_buffer[i]);
        }
    }

    const uint8_t *verify(const uint8_t *data, const uint8_t *end) {
        // We have two state lists, which we switch between a la double buffering.
        VE *states = _state_buffer[0], *next_states = _state_buffer[1];
        uint32_t _n_states[2];
        uint32_t *n_states = &_n_states[0], *next_n_states = &_n_states[1];

        // Also keep an index for each state in the list, pointing to where
        // in the input stream the match started
        uint32_t *start_idx = _start_buffer[0], *next_start_idx = _start_buffer[1];

        // For continuous operation, we will have the INITIAL_STATE always in
        // the current state set. To do this easily, we just always keep it
        // in the first position, and only write to entries after the first.
        if (Opts::CONTINUOUS) {
            states[0] = INITIAL_STATE();
            next_states[0] = INITIAL_STATE();
        }

        const uint8_t *input_p = Opts::FORWARDS ? data : end;

        *n_states = 1;
        states[0] = INITIAL_STATE();
        start_idx[0] = input_p - data + (Opts::FORWARDS ? -1 : 1);

        do {
start:
            uint8_t c = *input_p;

            // CONTINUOUS mode has an implied initial state always at the 
            // beginning of the buffer
            if (Opts::CONTINUOUS) {
                *next_n_states = 1;
                next_start_idx[0] = input_p - data;
            } else
                *next_n_states = 0;

            // Add a (lossy) bitset of already examined states this iteration,
            // to skip duplicate states. We add the start state to the mask so
            // we don't have an extra branch on every state--we only have to
            // check for the start state inside the duplicate checking code.
            LossyBitset seen;
            seen.add(TARGET_STATE());

            for (uint32_t s = 0; s < *n_states; s++) {
                VE state = states[s];
                V state_vec = broadcast<VI, VE>(state);
                V start_idx_vec = broadcast<VI, uint32_t>(start_idx[s]);

                // Check for whether we've already added this state
                if (seen.might_contain(state)) {
                    // We found a match! We added the target state to the
                    // seen bitset so we only have to do this test in the
                    // slow path.
                    if (state == TARGET_STATE()) {
                        // Non-continuous case: we were just verifying that there was
                        // a match. Oh hey, looks like there was one.
                        if (!Opts::CONTINUOUS)
                            return input_p;

                        // Continuous case: pass the match off to the match
                        // handler and keep going

                        // Register the match. We have to do a bit of
                        // index math to make the start/end points line
                        // up correctly in both the forwards and backwards
                        // cases. Why would we be streaming backwards? Who
                        // the hell knows?
                        if (Opts::FORWARDS)
                            match_handler.handle_match(*input_file,
                                    data + start_idx[s] + 1, input_p - 1);
                        else
                            match_handler.handle_match(*input_file,
                                    input_p + 1, data + start_idx[s] - 1);

                        // Skip to the next newline if we only care about
                        // one match per line
                        if (Opts::ONE_PER_LINE) {
                            input_p = skip_chr(input_p, '\n', Opts::FORWARDS,
                                    Opts::FORWARDS ? end : data);

                            next_start_idx[0] = input_p - data;

                            // Reset all states for the next iteration except
                            // the initial state (always at slot 0). Then
                            // break to go on to the next character.
                            *next_n_states = 1;
                            break;
                        }

                        // If we allow overlapping matches, continue on to
                        // the next state--all states that are in flight
                        // are still valid and need to be processed.
                        if (Opts::OVERLAPPING)
                            continue;

                        // No overlapping: any match that started before
                        // this character can be thrown away. *But* we need
                        // to process this character again from the start
                        // state. So reset the states to just the initial
                        // state, and go to the start
                        // XXX this is ugly
                        *n_states = 1;
                        goto start;
                    }

                    // We potentially have a duplicate state. Since the state could
                    // be anywhere in the state array before this state, we do a bulk
                    // compare using vector instructions against the whole array. In
                    // the loop we only compare up to the last part of the array that
                    // fits entirely within a vector. The rest are compared with a
                    // special case after.
                    uint32_t s_rounded = s & -N_VE;
                    for (uint32_t i = 0; i < s_rounded; i += N_VE) {
                        V key = *(V *)&states[i];
                        vmask eq = test_eq<VI, VE>(key, state_vec);
                        if (eq)
                            goto skip;
                    }

                    // Compare the last set of states before this one. We
                    // compare all of them, but only test the bits below
                    // the one corresponding to this state.
                    V key = *(V *)&states[s_rounded];
                    vmask eq = test_eq<VI, VE>(key, state_vec);
                    if (eq & (1 << (s - s_rounded)) - 1)
                        goto skip;
                } else
                    seen.add(state);

                for (uint32_t i = 0; i < key_count[c]; i += N_VE) {
                    // Load VL contiguous key bytes into one vector, and
                    // compare the current state against all of them at once
                    V key = *(V *)&edge_keys[c][i];
                    vmask eq = test_eq<VI, VE>(key, state_vec);

                    // If there was a match, there might be multiple
                    // predecessor states from this state/input byte. Since
                    // we sort the keys, we can just get the index of the
                    // first index with a bitscan, find the last matches with
                    // more vcmps and another bitscan, and copy all the bytes
                    // in between at once.
                    if (eq) {
                        uint32_t start = bsf64(eq) / MOVEMASK_HACK + i;

                        // Now that we have the start index, find the next
                        // key that *isn't* equal to this state. We can
                        // check for one within the same vector of keys that
                        // the first key was in with a simple bitwise check:
                        // all of the equal bits will be in one contiguous
                        // group. If we add in the least significant bit of
                        // the equality mask, we'll get a carry into the first
                        // zero bit. If there aren't any unequal keys within
                        // this vector after the equal keys, this addition
                        // will overflow past the bits in the VCMP_BITS mask.
                        vmask gt = eq + (eq & -eq);
                        gt &= VCMP_BITS;

                        // Loop through the rest of the keys until we find
                        // the first unequal key. We always store at least one
                        // sentinel at the end, so we don't need to check the
                        // length here
                        uint32_t j = i;
                        for (; !gt; j += N_VE) {
                            V key = *(V *)&edge_keys[c][j + N_VE];
                            gt = ~test_eq<VI, VE>(key, state_vec);
                        }

                        uint32_t end = bsf64(gt) / MOVEMASK_HACK + j;

                        // Copy an entire vectors' worth of values at a time,
                        // possibly past the end of the array (we should have
                        // enough space). We additionally copy over this state's
                        // starting index, propagating it to all of its next states.
                        VE *from = &edge_values[c][start];
                        VE *to = &next_states[*next_n_states];
                        uint32_t *to_idx = &next_start_idx[*next_n_states];
                        for (uint32_t x = start; x < end;
                                x += N_VE, from += N_VE, to += N_VE) {
                            *(V *)to = *(V *)from;

                            // Weird loop thing: if our main NFA states are only
                            // stored in one or two bytes, it takes more vector writes
                            // to copy our start index over to all the next states
                            for (uint32_t x_i = 0; x_i < sizeof(uint32_t) / sizeof(VE);
                                    x_i++, to_idx += (VL / sizeof(uint32_t)))
                                *(V *)to_idx = start_idx_vec;
                        }

                        // Update the index to reflect only the valid values
                        (*next_n_states) += end - start;
                        break;
                    }
                }
skip:           ;
            }

            // Flip the double buffers for the next iteration
            std::swap(states, next_states);
            std::swap(n_states, next_n_states);
            std::swap(start_idx, next_start_idx);
        } while (*n_states && 
                (Opts::FORWARDS ? ++input_p <= end : --input_p >= data));

        return NULL;
    }

    // Hacky method to run as a full matcher, not just verifier
    typename MatchHandler::return_type run(File &f) {
        match_handler.start();

        assert(Opts::CONTINUOUS);

        input_file = &f;
        verify(f.data, f.data + f.size - 1);

        return match_handler.finish(f);
    }
};

typedef _Squamatus<VEC_INFO, uint8_t, VerifierRegexOpts, DummyMatchHandler> SquamatusVerifier8;
typedef _Squamatus<VEC_INFO, uint16_t, VerifierRegexOpts, DummyMatchHandler> SquamatusVerifier16;
typedef _Squamatus<VEC_INFO, uint32_t, VerifierRegexOpts, DummyMatchHandler> SquamatusVerifier32;

////////////////////////////////////////////////////////////////////////////////
// Match handlers
// These classes take verified matches and perform an action (count it, print
// it out, etc.)
////////////////////////////////////////////////////////////////////////////////

// Count all occurrences of a match, ignoring newlines
class MatchHandlerBasicCounter {
    uint32_t _match_count;

public:
    typedef uint32_t return_type;

    MatchHandlerBasicCounter() { }

    void start() {
        _match_count = 0;
    }

    void handle_match(UNUSED File &f, UNUSED const uint8_t *start, UNUSED const uint8_t *end) {
        _match_count++;
    }

    return_type finish(UNUSED File &f) {
        return _match_count;
    }
};

// Basic grep-style handling: print out the full line, ignore other matches
// on the same line. We support counting matching lines and printing the path.
class MatchHandlerPrintLine {
    bool print_path;
    bool print_matches;
    bool print_count;
    bool print_colors;
    uint32_t match_count;
    const uint8_t *current_match_end;
    const uint8_t *current_line_end;

public:
    typedef uint32_t return_type;

    MatchHandlerPrintLine(bool print_path, bool print_matches, bool print_count,
            bool print_colors) : print_path(print_path), print_matches(print_matches),
            print_count(print_count), print_colors(print_colors) { }

    void start() {
        match_count = 0;
        current_match_end = NULL;
        current_line_end = NULL;
    }

    inline void flush_line() {
        if (print_matches && current_match_end) {
            std::string ending(current_match_end + 1, current_line_end);
            std::cout << ending << "\n";
            current_match_end = NULL;
        }
    }

    inline void handle_match(File &f, const uint8_t *start, const uint8_t *end) {
        const uint8_t *pre_match;

        // Is this match on a new line? If so, we possibly need to flush the last
        // part of the last match, and then print the beginning of this line
        // XXX this doesn't work for the backwards matching case, which
        // REALLY REALLY MATTERS
        if (start >= current_line_end) {
            flush_line();

            pre_match = skip_chr(start, '\n', false, f.data);
            current_line_end = skip_chr(end, '\n', true, f.data + f.size);
            if (print_path)
                std::cout << f.path << ":";
        }
        // Otherwise, we print out everything after the end of the last match
        // first
        else {
            pre_match = current_match_end;
            // Weird overlapping case--only happens when we allow overlapping
            // matches
            if (pre_match >= start)
                start = pre_match + 1;
        }

        // Print out everything on this line (or after the last match) leading
        // up to this match, and then print this match, optionally with ANSI
        // color highlighting
        if (print_matches) {
            // These strings have additional copies that are kind of annoying
            std::string begin(pre_match + 1, start);
            std::string mid(start, end + 1);

            if (print_colors)
                std::cout << begin << "\033[31;40m" << mid
                    << "\033[0m";
            else
                std::cout << begin << mid;
        }

        match_count++;

        // Store where the last match ended. We will print the rest of the line
        // later, either with more matches or not, and need to know where we
        // stopped printing
        current_match_end = end;
    }

    return_type finish(File &f) {
        flush_line();

        if (print_count) {
            if (print_path)
                std::cout << f.path << ":";
            std::cout << match_count << "\n";
        }
        return match_count;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Miroslav: the core SIMD string matching algorithm
////////////////////////////////////////////////////////////////////////////////
template<typename VI, typename MatchVerifier, typename MatchHandler,
    const uint32_t N_BYTES, const uint32_t N_SHIFTS, const uint32_t SHIFTS[]>
class _Miroslav {
    // Hacky substitute for "using"
    typedef typename VI::V V;
    typedef typename VI::vmask vmask;
    typedef typename VI::double_vmask double_vmask;
    static const uint32_t VL = VI::VL;
    static const uint32_t LMASK = VI::LMASK;

    static inline uint8_t state_mask(uint32_t state, uint32_t byte) {
        // If we only have one byte of state bits, start and end states
        // get the bottom and top bits, and all the other states rotate
        // through the other 6 bits
        if (N_BYTES == 1) {
            if (state == START_STATE)
                return 1 << 0;
            if (state == END_STATE)
                return 1 << 7;
            return 1 << (state % 6 + 1);
        }
        // For two or more state bytes, we use the top bit of the first
        // two bytes for the start and end states (for vpmovmskb convenience)
        // and all the other states rotate through the remaining bits
        else {
            if (state == START_STATE)
                return byte == 0 ? 1 << 7 : 0;
            if (state == END_STATE)
                return byte == 1 ? 1 << 7 : 0;
            state = state % (N_BYTES * 8 - 2);
            // Skip over bit 7 (start) and 15 (end)
            state += (state >= 7);
            state += (state >= 15);
            // Return the proper bit within this byte if the high bits of the
            // state id match the byte number
            return ((state >> 3) == byte) << (state & 7);
        }
    }

    static inline uint8_t char_mask(uint8_t c, uint32_t shift) {
        return (c >> shift) & LMASK;
    }

    MatchVerifier match_verifier;
    MatchHandler &match_handler;

    V from_states[N_BYTES][N_SHIFTS];
    V to_states[N_BYTES][N_SHIFTS];
    V v_char_mask;

    // For branchless testing of whether a pattern has a 1-character match.
    // Cuts down on false matches when the start and end characters of a
    // pattern hash to the same character class, which would cause every
    // occurrence of a character in that class to generate a match
    vmask has_1_char_match;

public:
    _Miroslav(NFAEdgeList &edges, MatchHandler &handler) :
            match_verifier(edges), match_handler(handler) {
        uint8_t from_state_bytes[N_BYTES][N_SHIFTS][VL] = {{{0}}};
        uint8_t to_state_bytes[N_BYTES][N_SHIFTS][VL] = {{{0}}};

        has_1_char_match = 0;

        // Initialize from/to state masks tables
        uint8_t c;
        uint32_t from, to;
        FOR_EACH_EDGE(c, from, to, edges) {
            for (uint32_t b = 0; b < N_BYTES; b++) {
                uint8_t fm = state_mask(from, b);
                uint8_t tm = state_mask(to, b);
                for (uint32_t s = 0; s < N_SHIFTS; s++) {
                    from_state_bytes[b][s][char_mask(c, SHIFTS[s])] |= fm;
                    to_state_bytes[b][s][char_mask(c, SHIFTS[s])] |= tm;
                }
            }

            // Check for 1-character matches and update the mask
            if (from == START_STATE && to == END_STATE)
                has_1_char_match = (vmask)-1;
        }

        for (uint32_t b = 0; b < N_BYTES; b++) {
            for (uint32_t s = 0; s < N_SHIFTS; s++) {
                VI::prepare_state_table(from_state_bytes[b][s]);
                VI::prepare_state_table(to_state_bytes[b][s]);
                from_states[b][s] = *(V*)from_state_bytes[b][s];
                to_states[b][s] = *(V*)to_state_bytes[b][s];
            }
        }

        v_char_mask = broadcast<VI, uint8_t>(LMASK);
    }

    typename MatchHandler::return_type run(File &f) {
        match_handler.start();

        double_vmask carry, last_carry = 0;

        // Fill a vector for each byte of state mask with the starting state. This
        // vector tracks the state between iterations of the main loop. Only the
        // last byte of each of these vectors is ever used.
        V last_to[N_BYTES];
        for (uint32_t b = 0; b < N_BYTES; b++)
            last_to[b] = broadcast<VI, uint8_t>(state_mask(START_STATE, b));

        const uint8_t *chunk;
        for (chunk = f.data; chunk + VL <= f.data + f.size; chunk += VL) {
            V input = *(V*)chunk;

            vmask start_m, end_m, seq_m = 0;

            for (uint32_t b = 0; b < N_BYTES; b++) {
                V from, to;
                // For each of the shifts defined in the template arguments, do
                // a vector table lookup by shifting each byte by the shift, masking,
                // and doing a permute on the appropriate table vector. We AND the
                // results together for each of these shifts, which should narrow down
                // the number of false positives from different input bytes
                // mapping to the same character class.
                for (uint32_t s = 0; s < N_SHIFTS; s++) {
                    V masked_input = input;
                    if (SHIFTS[s])
                        masked_input = VI::vec_shr(masked_input, SHIFTS[s]);
                    masked_input = VI::vec_and(masked_input, v_char_mask);

                    V f = VI::permute(from_states[b][s], masked_input);
                    V t = VI::permute(to_states[b][s], masked_input);
                    if (s == 0) {
                        from = f;
                        to = t;
                    } else {
                        from = VI::vec_and(from, f);
                        to = VI::vec_and(to, t);
                    }
                }

                // Get a vector of the to states, but shifted back in the data
                // stream by 1 byte. We fill the empty space in the first lane with
                // the last lane from last_to (which is initialized to the starting
                // state).
                V shl_to_1 = VI::vec_lanes_shl_1(to, last_to[b]);
                last_to[b] = to;

                // Test which input bytes can lead from the start state, and lead to
                // the end state. We handle the N_BYTES=1 and N cases differently,
                // because for 1, both bits are in the same byte, but otherwise, it's
                // a bit cheaper to have the start and end state bits in the top bit
                // of the first two state bytes, since vpmovmskb only looks at the
                // high bits of each byte in a vector. All this code should be 
                // unrolled/branchless, with start_m and end_m being set exactly once.
                if (N_BYTES == 1) {
                    start_m = VI::test_low_bit(from);
                    end_m = VI::test_high_bit(to);
                } else {
                    if (b == 0)
                        start_m = VI::test_high_bit(from);
                    else if (b == 1)
                        end_m = VI::test_high_bit(to);
                }

                // Now find all input bytes that can come from some state that the
                // previous input byte could lead to.
                V seq = VI::vec_and(shl_to_1, from);
                seq_m |= VI::test_nz(seq);
            }

            // Test for potential matches. We use the ripple of carries through the
            // "seq" mask to find sequences of input bytes that lead from the start
            // state to the end state, while passing through valid state transitions.
            // To be precise, since carries will ripple past the end mask, we find
            // bits in the end mask that are cleared by a carry. This is slightly
            // complicated by a couple factors: first, we have to keep track of
            // carries across iterations (which is why we use the "double_vmask"
            // type and shift carry right by VL each iteration), and second, extra
            // bits in the start mask can make us think a carry didn't happen, so
            // we clear out bits from start_m before testing the carries with
            // the end mask.
            carry = last_carry + ((double_vmask)start_m << 1) + seq_m;
            vmask covered = end_m & seq_m;
            vmask matches = ~(carry & ~start_m) & covered;

            // Check for 1-char matches, if they're possible
            matches |= has_1_char_match & start_m & end_m;

            last_carry = carry >> VL;

            // Look through the bitset of all potential matches, and run a
            // backwards verification step to weed out false positives. Any
            // real matches we pass off to the match handler.
            while (EXPECT(matches, 0)) {
                const uint8_t *end = chunk + bsf64(matches);
                matches &= matches - 1;

                const uint8_t *start = match_verifier.verify(f.data, end);
                if (EXPECT(start != NULL, 0))
                    match_handler.handle_match(f, start, end);
            }
        }

        // Run the slow backwards NFA on all the remainder bytes that don't fit
        // in a vector register. We could probably read past the original
        // buffer or do an unaligned read or something, but oh well.
        for (const uint8_t *end = chunk; end < f.data + f.size; end++) {
            const uint8_t *start = match_verifier.verify(f.data, end);
            if (start)
                match_handler.handle_match(f, start, end);
        }

        return match_handler.finish(f);
    }
};

// Template alias using VEC_INFO class defined in the Makefile
static constexpr uint32_t SHIFTS[] = {0, 3};
static constexpr uint32_t N_SHIFTS = sizeof(SHIFTS) / sizeof(SHIFTS[0]);
template <typename MatchHandler, typename MatchVerifier>
using Miroslav = _Miroslav<VEC_INFO, MatchVerifier, MatchHandler, 1, N_SHIFTS, SHIFTS>;
