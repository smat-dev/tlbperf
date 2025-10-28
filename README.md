# The Invisible Killer: How TLB Misses Wreck Perf

Everyone knows about cache misses. You profile your code, see L3 misses spiking, and immediately understand why performance tanked. But there's another memory hierarchy lurking beneath the surface that can silently destroy your performance: virtual memory translation.

The Translation Lookaside Buffer (TLB) and the page table machinery that implements virtual memory work beautifully most of the time but can easily turn fast memory accesses into multi-hundred-cycle stalls. If you haven't spent much time in the systems programming trenches you might find the extent of the effect surprising. Everything here applies to x86, ARM, and most flavors of GPU.

## A Simple Graph Walk

Let's consider a toy workload that demonstrates this beautifully. A graph walk: we have an array of vertices, where each vertex occupies exactly 64 bytes (one x86 cache line) and contains four indices pointing to outgoing edges. Our algorithm is dead simple - land on a vertex, randomly pick one of the four edges, follow it to the next vertex, repeat.

When the entire graph fits in cache, this flies. For larger datasets you would think that since each vertex is just a single load, we would be looking at ~70-80 ns. But scale up to a 10 GB working set and something strange happens: the per-hop time starts showing triple digit nanoseconds and p95 shoots up to 2x the median time. So what's burning the extra cycles?

Plot the distribution of per-hop times as you scale and you see three distinct regimes emerge. At small sizes everything is straight from cache. Scale to hundreds of megabytes and times increase but plateau - loads are hitting DRAM, but TLB entries are still cached. But at multiple gigabytes suddenly you're in a world of pain: full page table walks.

![Per-hop latency quantiles versus working-set size with 4 KiB pages](figures/baseline_standard_quantiles.png)

## The Hidden Machinery

Quick refresher: modern processors don't directly access physical memory addresses. Every memory reference goes through virtual memory translation - your program's virtual address gets mapped to a physical address via page tables managed by the kernel. The TLB is a small, fast cache of recent translations. On x86 you typically get 64 L1 dTLB 4 KiB entries per core; the unified L2 or shared TLB has about 1.5K entries on Intel and 2-3K on newer AMD parts.

This is where it gets expensive. When the TLB misses, the processor must perform a page table walk: it traverses a multi-level tree structure stored in memory to find the translation. On x86-64 with 4-level paging this walk visits PML4 (bits 47-39), PDPT (bits 38-30), PD (bits 29-21), and PT (bits 20-12), then combines the physical frame with the offset (bits 11-0) to produce a roughly 48 bit physical address. On processors with 5-level paging support, an additional PML5 level extends the address space and adds another hop for about 8 more bits. Each level requires a memory access to read the entry that points to the next level.

Think about the worst case: four serial memory accesses, each potentially taking 70-80 ns if they miss all caches, just to figure out where your data lives. Only then can you load the actual data. In practice, upper-level page table entries often hit in L3 or in dedicated paging structure caches, so walk costs range from near-L3 latencies (20-30 ns) to full DRAM trips. But when your working set thrashes those caches you're staring at the full several hundred nanosecond worst case. Your fast processor sits idle waiting on one distant memory request after another.

Making matters worse, page table walkers are scarce resources - typically just a few per core (Intel Skylake and newer have two; AMD Zen 4 has six). When they're exhausted, memory loads cannot even begin their walk. Instructions pile up in the reorder buffer and your processor grinds to a halt.

You can make this visible with hardware performance counters. This is processor and platform specific but as a starting point on Linux plus Intel:

```
perf stat -e dTLB-loads,dTLB-load-misses,dtlb_load_misses.miss_causes_a_walk,dtlb_load_misses.walk_completed,dtlb_load_misses.walk_duration
```

On our graph walk with a 10 GB working set you will see millions of TLB misses and walks.

![dTLB-load-misses per hop across page sizes](figures/tlb_misses.png)

Virtualization makes everything exponentially worse. In a VM you have two-dimensional paging: guest virtual addresses translate to guest physical addresses (via guest page tables), which then translate to host physical addresses (via host page tables). In the worst case with EPT or NPT this can require up to 24 memory references when nothing is cached - every step of the guest walk may require a full host walk. Modern hardware and hypervisors use a bag of tricks to pull the practical cost down to under 2-3x native, but they cannot eliminate the fundamental overhead.

Brief aside on NUMA: on multi-socket systems accessing remote memory adds significant penalty - often about 1.5x local latency. Page table walks on remote memory are best avoided if possible.

## Fighting Back: Optimization Strategies

Use containers, not VMs. Containers on bare metal avoid the nested translation overhead entirely. The performance difference for memory-intensive workloads can be dramatic.

Huge pages are your friend. x86 supports 4 KB, 2 MB, and 1 GB pages (ARM has similar options). Larger pages have multiple benefits: they reduce page table depth (2 MB pages skip the bottom level), massively increase TLB reach (one entry covers 512x more memory for 2 MB pages), and keep more page table entries in cache. The key thing to realize is that these effects are multiplicative - shorter walks on fewer misses, with much faster access to page table entries.

The catch is granularity. The OS allocates memory in page-sized chunks. 1 GB pages mean 1 GB allocations - fine for a massive in-memory database, terrible for general-purpose computing. Swapping to disk becomes absurd: paging out 1 GB to access a few bytes would absolutely cripple performance. Additionally OSes often require explicit pinning at boot or through hugetlbfs due to physical memory fragmentation. 2 MiB pages are much more practical, but still tend to require some setup on the OS side.

![Latency quantiles for the pointer chase with 2 MiB pages](figures/baseline_2m_quantiles.png)

Most processors have split TLB hierarchies with fewer entries for huge pages. For some access patterns this can actually increase TLB misses, but if it means page tables fit in cache and walks are fast it usually ends up being worth the tradeoff.

Linux transparent hugepages (THP) tries to automatically promote 4 KB to 2 MB pages in the background. This works brilliantly for sequential access patterns, but for random access workloads it can be somewhere between useless and actively counterproductive because of the background work involved.

If you can adjust your data access patterns, try TLB blocking - exactly the same principle as cache blocking but at a higher level. Structure your algorithm to maintain high locality within blocks that can exceed cache size but fit in the TLB's reach (either directly or via fast walks to cached page tables). With our graph walk, if we process vertices within blocks we maintain spatial locality at the page level even when cache thrashes. The TLB miss regime virtually disappears, as the comparison below makes clear.

![Latency quantiles comparing baseline, huge pages, and TLB blocking strategies](figures/baseline_quantiles.png)

Software prefetching can help with pointer chasing if you can predict the next address early, because the prefetch may initiate translation (very CPU dependent).

## The Math: A Simple Model

Model it simply: let the working set span M pages with TLB capacity C pages. Under uniform random access with LRU replacement, miss probability rises sharply once M >> C, with the knee around M ~= C.

The cost per access is: `T ~= T_hit + p_miss * T_walk`.

Huge pages decrease both p_miss (more coverage per entry) and T_walk (fewer levels). This multiplicative benefit makes them incredibly effective. T_walk also explodes when page table entries miss L3 and you're walking through main memory - easily 4-5x worse than when everything is cached.

## Putting It Together

The benchmark code for the graph walk is available on GitHub (<https://github.com/smat-dev/tlbperf>). We allocate arrays of varying sizes, perform millions of random hops, and measure per-hop latency at different percentiles.

![Mitigation strategies compared at a large working set](figures/mitigation_bars.png)

Quick note on GPUs: the same concepts apply - GPU virtual memory also uses TLBs and page tables - but implementation details differ significantly. Modern GPUs have larger TLBs and different page table formats optimized for their access patterns.

Takeaway: virtual memory translation is a heavyweight abstraction you cannot opt out of. It is worth knowing the costs and how to manage them. For large workloads with poor locality, TLB misses can dominate performance.
