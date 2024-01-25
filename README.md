# Generalised Isotonic Regression

## Goal

This library is an attempt at an open source implementation of [Generalised Isotonic Regression](https://arxiv.org/abs/1104.1779) by
Ronny Luss & Saharon Rosset. They have provided some Matlab code [here](https://www.tandfonline.com/doi/suppl/10.1080/10618600.2012.741550?scroll=top)
that additionally requires a [Mosek](https://www.mosek.com/) license.

Isotonic Regression tries to fit a line/plane/hyperplane to a sequence of observations that lies as "close" as possible
to the observations, while maintaining monotonicity.

The generalised variant presented in [Generalised Isotonic Regression](https://arxiv.org/abs/1104.1779) supports
convex loss functions in N-dimensions.

## Building CLI Tool

Requires

* C++ 17 compatible Clang
* CMake version 3.16

```bash
git clone https://github.com/ewal31/GeneralisedIsotonicRegression
cd GeneralisedIsotonicRegression
cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_GIR_CLI_TOOL=ON
make -C build
```

The built cli tool can then be found at `build/cli/gir`

## CLI Tool Example

```
Usage:
  gir [OPTION...]

  -i, --input arg         Input File
  -o, --output arg        Output File
  -l, --loss arg          Loss Function [L1|L2|L2_WEIGHTED|HUBER] (default:
                          L2)
      --delta arg         Huber Loss Delta (default: 1.0)
  -m, --monotonicity arg  Comma separated monotonicity direction for each
                          column of X: '1' for ascending, '-1' for
                          descending. (default: ascending)
  -h, --help              Print usage
```

Create a file `input.csv` with the following contents.

```
X_1,       X_2,       y,         weight
0.762495,  0.392963,  0.301295,  1
0.522416,  0.486052,  0.254428,  1
0.796809,  0.0152406, 0.0120925, 1
0.979002,  0.271266,  0.265759,  1
0.0711248, 0.310313,  0.9222442, 0.1
```

* The total number of X dimensions (X\_1, X\_2, ...) is variable.
* Including weights is optional.
* The columns can have any arbitrary ordering.

Traditional Isotonic Regression with the L\_2 norm can then be run as follows

```bash
gir -i input.csv -o output.csv
```

or to use a weighted L\_2 norm

```bash
gir -l L2_WEIGHTED -i input.csv -o output.csv
```

alternatively, to specificy monotonicity should be maintained in the opposite
direction

```bash
gir -i input.csv -o output.csv -m -1,-1
```

## Library

The library can be included in CMake projects by including the following snippet.

```CMake
include(FetchContent)

FetchContent_Declare(
    GIR
    GIT_REPOSITORY "https://github.com/ewal31/GeneralisedIsotonicRegression"
    GIT_TAG 0.2.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    GIR
)

target_link_libraries(
    <Target-Name>
    PRIVATE
    isotonic_regression
)
```

The main functions of interest are then

```cpp
template<typename V>
std::tuple<Eigen::SparseMatrix<bool>, VectorXu, VectorXu>
points_to_adjacency(
    const Eigen::MatrixX<V>& points
)
```

and

```cpp
std::pair<VectorXu, Eigen::VectorXd>
generalised_isotonic_regression (
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    YType&& _y,
    WeightsType&& _weights,
    const LossFunction<LossType>& loss_fun,
    uint64_t max_iterations = 0
)
```

An example can be found in [example/main.cpp](./example/main.cpp).

## Dependencies

* [Eigen](https://gitlab.com/libeigen/eigen)
* [HiGHS](https://github.com/ERGO-Code/HiGHS)

### For CLI Tool

* [cxxopts](https://github.com/jarro2783/cxxopts)
* [Vince's CSV Parser](https://github.com/vincentlaucsb/csv-parser)

### For Testing

* [Catch2](https://github.com/catchorg/Catch2)

## TODO

- [x] rewrite loss functions code to allow for more to be added and so they can be configured
- [x] tests for duplicate points
- [x] more loss functions
- [ ] tests for really small weights or weights of 0
- [x] configurable direction for monotonicity in cli (just need to multiply by -ve 1 right?)
- [ ] rewrite `points_to_adjacency` function to be more efficient
    - [x] 1d
    - [ ] 2d
    - [ ] 3 and more dimensions
- [ ] compare more thoroughly with other isotonic regression implementations
- [ ] add more information for users of cli tool
    - [ ] optional progress bar for cli would be nice
    - [ ] option for saving all iterative solutions
- [ ] add Python interface
