#include <benchmark/benchmark.h>

#include <generalized_isotonic_regression.h>

// static void BM_points_to_adjacency_1d(benchmark::State& state) {
//     for (auto _ : state) {
//         state.PauseTiming();
//         const auto [points, y] = gir::generate_monotonic_points(state.range(0), 0.1, 1);
//         state.ResumeTiming();
//         benchmark::DoNotOptimize(gir::points_to_adjacency_1d(points));
//         state.SetComplexityN(state.range(0));
//     }
// }
// BENCHMARK(BM_points_to_adjacency_1d)
//     ->RangeMultiplier(2)->Range(1<<10, 1<<16)->Complexity();
// 
// static void BM_points_to_adjacency_1_brute_force(benchmark::State& state) {
//     for (auto _ : state) {
//         state.PauseTiming();
//         const auto [points, y] = gir::generate_monotonic_points(state.range(0), 0.1, 1);
//         state.ResumeTiming();
//         benchmark::DoNotOptimize(gir::points_to_adjacency_N_brute_force(points));
//         state.SetComplexityN(state.range(0));
//     }
// }
// BENCHMARK(BM_points_to_adjacency_1_brute_force)
//     ->RangeMultiplier(2)->Range(1<<10, 1<<16)->Complexity();

static void BM_points_to_adjacency_2d(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        const auto [points, y] = gir::generate_monotonic_points(state.range(0), 0.1, 2);
        state.ResumeTiming();
        benchmark::DoNotOptimize(gir::points_to_adjacency_2d(points));
        state.SetComplexityN(state.range(0));
    }
}
BENCHMARK(BM_points_to_adjacency_2d)
    ->RangeMultiplier(2)->Range(1<<10, 1<<16)->Complexity();

static void BM_points_to_adjacency_2d_divide_and_conquer(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        const auto [points, y] = gir::generate_monotonic_points(state.range(0), 0.1, 2);
        state.ResumeTiming();
        benchmark::DoNotOptimize(gir::points_to_adjacency_2d_divide_and_conquer(points));
        state.SetComplexityN(state.range(0));
    }
}
BENCHMARK(BM_points_to_adjacency_2d_divide_and_conquer)
    ->RangeMultiplier(2)->Range(1<<10, 1<<16)->Complexity();

static void BM_points_to_adjacency_2_brute_force(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        const auto [points, y] = gir::generate_monotonic_points(state.range(0), 0.1, 2);
        state.ResumeTiming();
        benchmark::DoNotOptimize(gir::points_to_adjacency_N_brute_force(points));
        state.SetComplexityN(state.range(0));
    }
}
BENCHMARK(BM_points_to_adjacency_2_brute_force)
    ->RangeMultiplier(2)->Range(1<<10, 1<<16)->Complexity();

// static void BM_points_to_adjacency_3_brute_force(benchmark::State& state) {
//     for (auto _ : state) {
//         state.PauseTiming();
//         const auto [points, y] = gir::generate_monotonic_points(state.range(0), 0.1, 3);
//         state.ResumeTiming();
//         benchmark::DoNotOptimize(gir::points_to_adjacency_N_brute_force(points));
//         state.SetComplexityN(state.range(0));
//     }
// }
// BENCHMARK(BM_points_to_adjacency_3_brute_force)
//     ->RangeMultiplier(2)->Range(1<<10, 1<<16)->Complexity();

BENCHMARK_MAIN();
