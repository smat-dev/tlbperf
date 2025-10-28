#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <xmmintrin.h>
#include <x86intrin.h>
#ifdef __linux__
#include <sys/prctl.h>
#include <linux/prctl.h>
#ifndef PR_TASK_PERF_EVENTS_DISABLE
#define PR_TASK_PERF_EVENTS_DISABLE 0x326
#endif
#ifndef PR_TASK_PERF_EVENTS_ENABLE
#define PR_TASK_PERF_EVENTS_ENABLE 0x327
#endif
#ifndef PR_TASK_PERF_EVENTS_RESET
#define PR_TASK_PERF_EVENTS_RESET 0x327 + 1
#endif
#endif

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

namespace {

enum class Mode { kBaseline, kBlocked, kPrefetch };

struct Options {
    size_t num_vertices = 1 << 20;
    size_t steps = 1 << 20;
    Mode mode = Mode::kBaseline;
    size_t block_vertices = 1 << 17;
    size_t block_steps = 1 << 18;
    size_t warmup_steps = 10000;
    int prefetch_distance = 1;
    bool advise_huge = false;
    bool advise_nohuge = false;
    bool use_hugetlb_2m = false;
    bool use_hugetlb_1g = false;
    std::string output_path;
    unsigned seed = 5489u;
    bool warmup_overridden = false;
};

struct alignas(64) Vertex {
    std::array<uint32_t, 4> edges{};
};

struct VertexBuffer {
    VertexBuffer() = default;
    VertexBuffer(const VertexBuffer&) = delete;
    VertexBuffer& operator=(const VertexBuffer&) = delete;
    VertexBuffer(VertexBuffer&& other) noexcept { move_from(std::move(other)); }
    VertexBuffer& operator=(VertexBuffer&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }
    ~VertexBuffer() { release(); }

    void allocate(size_t count, const Options& opt) {
        release();
        count_ = count;
        if (count_ == 0) {
            return;
        }
        size_t bytes = count_ * sizeof(Vertex);
        size_t alignment = 4096;
        if (opt.use_hugetlb_1g) {
            alignment = 1ULL << 30;
        } else if (opt.use_hugetlb_2m) {
            alignment = 2ULL << 20;
        }
        if (opt.use_hugetlb_2m || opt.use_hugetlb_1g) {
            mapped_bytes_ = align_up(bytes, alignment);
            ensure_memlock_limit(mapped_bytes_);
            int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;
            flags |= opt.use_hugetlb_1g ? MAP_HUGE_1GB : MAP_HUGE_2MB;
            void* ptr = mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE, flags, -1, 0);
            if (ptr == MAP_FAILED) {
                int err = errno;
                throw std::runtime_error(std::string("mmap(MAP_HUGETLB) failed: ") + std::strerror(err));
            }
            data_ = static_cast<Vertex*>(ptr);
            use_mmap_ = true;
        } else {
            mapped_bytes_ = bytes;
            void* mem = nullptr;
            if (posix_memalign(&mem, alignment, mapped_bytes_) != 0) {
                throw std::runtime_error("posix_memalign failed");
            }
            data_ = static_cast<Vertex*>(mem);
            use_mmap_ = false;
        }
        std::memset(data_, 0, mapped_bytes_);
    }

    Vertex* data() { return data_; }
    const Vertex* data() const { return data_; }
    size_t size() const { return count_; }
    size_t mapped_bytes() const { return mapped_bytes_; }

private:
    static size_t align_up(size_t value, size_t alignment) {
        if (alignment == 0) {
            return value;
        }
        size_t remainder = value % alignment;
        if (remainder == 0) {
            return value;
        }
        return value + (alignment - remainder);
    }

    static void ensure_memlock_limit(size_t bytes) {
        if (bytes == 0) {
            return;
        }
        struct rlimit limit;
        if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) {
            std::perror("getrlimit(RLIMIT_MEMLOCK)");
            return;
        }
        rlim_t required = static_cast<rlim_t>(bytes);
        if (limit.rlim_cur >= required && limit.rlim_max >= required) {
            return;
        }
        struct rlimit new_limit = limit;
        if (new_limit.rlim_cur < required) {
            new_limit.rlim_cur = required;
        }
        if (new_limit.rlim_max < required) {
            new_limit.rlim_max = required;
        }
        if (setrlimit(RLIMIT_MEMLOCK, &new_limit) != 0) {
            std::perror("setrlimit(RLIMIT_MEMLOCK)");
        }
    }

    void release() {
        if (!data_) {
            return;
        }
        if (use_mmap_) {
            if (munmap(data_, mapped_bytes_) != 0) {
                std::perror("munmap");
            }
        } else {
            std::free(data_);
        }
        data_ = nullptr;
        count_ = 0;
        mapped_bytes_ = 0;
        use_mmap_ = false;
    }

    void move_from(VertexBuffer&& other) {
        data_ = other.data_;
        count_ = other.count_;
        mapped_bytes_ = other.mapped_bytes_;
        use_mmap_ = other.use_mmap_;
        other.data_ = nullptr;
        other.count_ = 0;
        other.mapped_bytes_ = 0;
        other.use_mmap_ = false;
    }

    Vertex* data_ = nullptr;
    size_t count_ = 0;
    size_t mapped_bytes_ = 0;
    bool use_mmap_ = false;
};
struct XorShift64 {
    explicit XorShift64(uint64_t s = 1) : state(s ? s : 0xdeadbeefULL) {}
    uint64_t operator()() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return state = x;
    }
    uint64_t state;
};

uint64_t rdtscp_now() {
    unsigned aux;
    return __rdtscp(&aux);
}

uint64_t measure_rdtscp_overhead() {
    uint64_t min_delta = UINT64_MAX;
    for (int i = 0; i < 1000; ++i) {
        uint64_t t0 = rdtscp_now();
        uint64_t t1 = rdtscp_now();
        min_delta = std::min(min_delta, t1 - t0);
    }
    return min_delta;
}

double read_cpu_mhz() {
    auto read_sysfs_khz = [](const char* path) -> double {
        std::ifstream f(path);
        if (!f) {
            return 0.0;
        }
        long long khz = 0;
        f >> khz;
        if (khz <= 0) {
            return 0.0;
        }
        return static_cast<double>(khz) / 1000.0;
    };

    // Prefer fixed frequency readings to avoid transient Cpufreq sampling.
    double mhz = read_sysfs_khz("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (mhz <= 0.0) {
        mhz = read_sysfs_khz("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency");
    }
    if (mhz > 0.0) {
        return mhz;
    }

    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo) {
        return 0.0;
    }
    std::string line;
    while (std::getline(cpuinfo, line)) {
        constexpr char prefix[] = "cpu MHz";
        if (line.compare(0, sizeof(prefix) - 1, prefix) == 0) {
            auto pos = line.find(':');
            if (pos == std::string::npos) {
                continue;
            }
            double mhz = std::atof(line.c_str() + pos + 1);
            if (mhz > 0.0) {
                return mhz;
            }
        }
    }
    return 0.0;
}

#ifdef __linux__
void perf_disable() {
#ifdef PR_TASK_PERF_EVENTS_DISABLE
    if (prctl(PR_TASK_PERF_EVENTS_DISABLE, 0) != 0) {
        std::perror("prctl(PR_TASK_PERF_EVENTS_DISABLE)");
    }
#endif
}

void perf_enable() {
#ifdef PR_TASK_PERF_EVENTS_RESET
    if (prctl(PR_TASK_PERF_EVENTS_RESET, 0) != 0) {
        std::perror("prctl(PR_TASK_PERF_EVENTS_RESET)");
    }
#endif
#ifdef PR_TASK_PERF_EVENTS_ENABLE
    if (prctl(PR_TASK_PERF_EVENTS_ENABLE, 0) != 0) {
        std::perror("prctl(PR_TASK_PERF_EVENTS_ENABLE)");
    }
#endif
}
#else
void perf_disable() {}
void perf_enable() {}
#endif

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto consume_value = [&](const std::string& prefix) -> std::string {
            if (arg.size() > prefix.size()) {
                return arg.substr(prefix.size());
            }
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + prefix);
            }
            return argv[++i];
        };
        if (arg.rfind("--size=", 0) == 0) {
            opt.num_vertices = std::stoull(consume_value("--size="));
        } else if (arg == "--size") {
            opt.num_vertices = std::stoull(consume_value("--size"));
        } else if (arg.rfind("--steps=", 0) == 0) {
            opt.steps = std::stoull(consume_value("--steps="));
        } else if (arg == "--steps") {
            opt.steps = std::stoull(consume_value("--steps"));
        } else if (arg.rfind("--mode=", 0) == 0) {
            auto value = consume_value("--mode=");
            if (value == "baseline") {
                opt.mode = Mode::kBaseline;
            } else if (value == "blocked") {
                opt.mode = Mode::kBlocked;
            } else if (value == "prefetch") {
                opt.mode = Mode::kPrefetch;
            } else {
                throw std::runtime_error("Unknown mode: " + value);
            }
        } else if (arg == "--mode") {
            auto value = consume_value("--mode");
            if (value == "baseline") {
                opt.mode = Mode::kBaseline;
            } else if (value == "blocked") {
                opt.mode = Mode::kBlocked;
            } else if (value == "prefetch") {
                opt.mode = Mode::kPrefetch;
            } else {
                throw std::runtime_error("Unknown mode: " + value);
            }
        } else if (arg.rfind("--block-vertices=", 0) == 0) {
            opt.block_vertices = std::stoull(consume_value("--block-vertices="));
        } else if (arg == "--block-vertices") {
            opt.block_vertices = std::stoull(consume_value("--block-vertices"));
        } else if (arg.rfind("--block-steps=", 0) == 0) {
            opt.block_steps = std::stoull(consume_value("--block-steps="));
        } else if (arg == "--block-steps") {
            opt.block_steps = std::stoull(consume_value("--block-steps"));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            opt.warmup_steps = std::stoull(consume_value("--warmup="));
            opt.warmup_overridden = true;
        } else if (arg == "--warmup") {
            opt.warmup_steps = std::stoull(consume_value("--warmup"));
            opt.warmup_overridden = true;
        } else if (arg.rfind("--prefetch-distance=", 0) == 0) {
            opt.prefetch_distance = std::stoi(consume_value("--prefetch-distance="));
        } else if (arg == "--prefetch-distance") {
            opt.prefetch_distance = std::stoi(consume_value("--prefetch-distance"));
        } else if (arg == "--advise-huge") {
            opt.advise_huge = true;
        } else if (arg == "--advise-nohuge") {
            opt.advise_nohuge = true;
        } else if (arg == "--hugetlb-2m") {
            opt.use_hugetlb_2m = true;
        } else if (arg == "--hugetlb-1g") {
            opt.use_hugetlb_1g = true;
        } else if (arg.rfind("--seed=", 0) == 0) {
            opt.seed = static_cast<unsigned>(std::stoul(consume_value("--seed=")));
        } else if (arg == "--seed") {
            opt.seed = static_cast<unsigned>(std::stoul(consume_value("--seed")));
        } else if (arg.rfind("--output=", 0) == 0) {
            opt.output_path = consume_value("--output=");
        } else if (arg == "--output") {
            opt.output_path = consume_value("--output");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opt.output_path.empty()) {
        throw std::runtime_error("Missing --output <path>");
    }
    if (opt.use_hugetlb_2m && opt.use_hugetlb_1g) {
        throw std::runtime_error("Choose at most one of --hugetlb-2m or --hugetlb-1g");
    }
    if ((opt.use_hugetlb_2m || opt.use_hugetlb_1g) && (opt.advise_huge || opt.advise_nohuge)) {
        throw std::runtime_error("Cannot combine hugetlb allocation with madvise hints");
    }
    if (opt.advise_huge && opt.advise_nohuge) {
        throw std::runtime_error("Choose at most one of --advise-huge or --advise-nohuge");
    }
    if (opt.prefetch_distance < 0) {
        opt.prefetch_distance = 0;
    }
    return opt;
}

void advise_memory(void* ptr, size_t bytes, const Options& opt) {
    if (opt.use_hugetlb_2m || opt.use_hugetlb_1g) {
        return;
    }
    if (opt.advise_huge) {
        if (madvise(ptr, bytes, MADV_HUGEPAGE) != 0) {
            std::perror("madvise(MADV_HUGEPAGE)");
        }
    } else if (opt.advise_nohuge) {
        if (madvise(ptr, bytes, MADV_NOHUGEPAGE) != 0) {
            std::perror("madvise(MADV_NOHUGEPAGE)");
        }
    }
}

void prefault_memory(Vertex* vertices, size_t n) {
    volatile uint32_t sink = 0;
    for (size_t i = 0; i < n; ++i) {
        sink ^= vertices[i].edges[0];
    }
    (void)sink;
}

void initialise_edges(Vertex* vertices, size_t n, Mode mode, size_t block_vertices, XorShift64& rng) {
    if (n == 0) {
        return;
    }
    if (mode == Mode::kBlocked) {
        const size_t block_mask = block_vertices - 1;
        if ((block_vertices & block_mask) != 0) {
            throw std::runtime_error("block_vertices must be a power of two for blocked mode");
        }
        for (size_t i = 0; i < n; ++i) {
            size_t block_base = (i / block_vertices) * block_vertices;
            for (auto& next : vertices[i].edges) {
                uint64_t r = rng();
                size_t offset = static_cast<size_t>(r & block_mask);
                next = static_cast<uint32_t>(block_base + offset);
            }
        }
    } else {
        const uint32_t mask = static_cast<uint32_t>(n - 1);
        bool is_power_of_two = (n & (n - 1)) == 0;
        for (size_t i = 0; i < n; ++i) {
            for (auto& next : vertices[i].edges) {
                uint64_t r = rng();
                if (is_power_of_two) {
                    next = static_cast<uint32_t>(r & mask);
                } else {
                    next = static_cast<uint32_t>(r % n);
                }
            }
        }
    }
}

void warmup_walk(const Vertex* vertices, size_t n, size_t steps, XorShift64& rng, Mode mode, size_t block_vertices, size_t block_steps) {
    if (steps == 0) {
        return;
    }
    uint32_t current = 0;
    if (n == 0) {
        return;
    }
    size_t block_base = 0;
    size_t block_countdown = block_steps;
    for (size_t i = 0; i < steps; ++i) {
        if (mode == Mode::kBlocked) {
            if (block_countdown == 0) {
                uint64_t r = rng();
                size_t block_idx = static_cast<size_t>(r % std::max<size_t>(1, n / block_vertices));
                block_base = block_idx * block_vertices;
                block_countdown = block_steps;
            }
            --block_countdown;
        }
        uint64_t r = rng();
        uint32_t choice = static_cast<uint32_t>(r & 3);
        uint32_t next = vertices[current].edges[choice];
        if (mode == Mode::kBlocked) {
            next = static_cast<uint32_t>(block_base + (next % block_vertices));
        }
        current = next % n;
    }
}

std::vector<uint32_t> run_walk(const Vertex* vertices, size_t n, const Options& opt, uint64_t overhead_cycles, double ns_per_cycle) {
    std::vector<uint32_t> per_hop(opt.steps);
    if (n == 0) {
        return per_hop;
    }
    XorShift64 rng(opt.seed * 0x9e3779b97f4a7c15ULL + 1);
    uint32_t current = 0;
    size_t block_base = 0;
    size_t block_countdown = opt.block_steps;
    size_t block_vertices = std::max<size_t>(1, opt.block_vertices);

    std::vector<uint32_t> upcoming(opt.prefetch_distance, 0);
    size_t upcoming_idx = 0;

    for (size_t step = 0; step < opt.steps; ++step) {
        if (opt.mode == Mode::kBlocked) {
            if (block_countdown == 0) {
                uint64_t r = rng();
                size_t block_idx = static_cast<size_t>(r % std::max<size_t>(1, n / block_vertices));
                block_base = block_idx * block_vertices;
                block_countdown = opt.block_steps;
            }
            --block_countdown;
        }

        uint64_t r = rng();
        uint32_t choice = static_cast<uint32_t>(r & 3);
        uint64_t t0 = rdtscp_now();
        uint32_t next = vertices[current].edges[choice];
        uint64_t t1 = rdtscp_now();
        uint64_t delta = (t1 - t0);
        if (delta > overhead_cycles) {
            delta -= overhead_cycles;
        } else {
            delta = 0;
        }

        if (opt.mode == Mode::kBlocked) {
            next = static_cast<uint32_t>(block_base + (next % block_vertices));
        }
        next %= n;

        if (opt.mode == Mode::kPrefetch && opt.prefetch_distance > 0) {
            upcoming[upcoming_idx] = next;
            size_t pf_idx = (upcoming_idx + 1) % upcoming.size();
            if (upcoming.size() > 1 || step >= opt.prefetch_distance) {
                _mm_prefetch(reinterpret_cast<const char*>(&vertices[upcoming[pf_idx]]), _MM_HINT_NTA);
            }
            upcoming_idx = pf_idx;
        }

        per_hop[step] = static_cast<uint32_t>(delta);
        current = next;
    }

    return per_hop;
}

void write_output(const Options& opt, const std::vector<uint32_t>& per_hop, double ns_per_cycle, uint64_t overhead_cycles) {
    std::ofstream out(opt.output_path, std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open output: " + opt.output_path);
    }
    auto mode_to_string = [](Mode m) -> std::string {
        switch (m) {
            case Mode::kBaseline: return "baseline";
            case Mode::kBlocked: return "blocked";
            case Mode::kPrefetch: return "prefetch";
        }
        return "baseline";
    };
    const char* hugetlb_mode = "none";
    if (opt.use_hugetlb_1g) {
        hugetlb_mode = "1g";
    } else if (opt.use_hugetlb_2m) {
        hugetlb_mode = "2m";
    }
    out << "# mode=" << mode_to_string(opt.mode)
        << " vertices=" << opt.num_vertices
        << " steps=" << opt.steps
        << " block_vertices=" << opt.block_vertices
        << " block_steps=" << opt.block_steps
        << " prefetch_distance=" << opt.prefetch_distance
        << " overhead_cycles=" << overhead_cycles
        << " ns_per_cycle=" << ns_per_cycle
        << " advise_huge=" << opt.advise_huge
        << " advise_nohuge=" << opt.advise_nohuge
        << " hugetlb_mode=" << hugetlb_mode
        << "\n";
    out << "index,cycles,nanoseconds\n";
    for (size_t i = 0; i < per_hop.size(); ++i) {
        double ns = per_hop[i] * ns_per_cycle;
        out << i << "," << per_hop[i] << "," << std::fixed << std::setprecision(3) << ns << "\n";
        out.unsetf(std::ios::floatfield);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        perf_disable();

        VertexBuffer vertices;
        vertices.allocate(opt.num_vertices, opt);
        XorShift64 rng(opt.seed);
        initialise_edges(vertices.data(), vertices.size(), opt.mode, opt.block_vertices, rng);
        advise_memory(vertices.data(), vertices.size() * sizeof(Vertex), opt);

        if (!opt.warmup_overridden) {
            size_t warmup_target = opt.steps;
            if (opt.num_vertices > 0) {
                const size_t limit = std::numeric_limits<size_t>::max() / 4;
                size_t scaled = opt.num_vertices > limit ? std::numeric_limits<size_t>::max()
                                                         : opt.num_vertices * 4;
                warmup_target = std::min(opt.steps, scaled);
            }
            if (warmup_target > opt.warmup_steps) {
                opt.warmup_steps = warmup_target;
            }
        }

        prefault_memory(vertices.data(), vertices.size());
        warmup_walk(vertices.data(), vertices.size(), opt.warmup_steps, rng, opt.mode, opt.block_vertices, opt.block_steps);

        perf_enable();

        uint64_t overhead = measure_rdtscp_overhead();
        double cpu_mhz = read_cpu_mhz();
        double ns_per_cycle = cpu_mhz > 0.0 ? (1000.0 / cpu_mhz) : 0.0;

        auto per_hop = run_walk(vertices.data(), vertices.size(), opt, overhead, ns_per_cycle);
        perf_disable();

        write_output(opt, per_hop, ns_per_cycle, overhead);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
