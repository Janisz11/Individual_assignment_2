#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <tuple>
#include <stdexcept>
#include <string>
#include <algorithm>

using Matrix = std::vector<double>;

inline double &at(Matrix &M, int n, int i, int j) {
    return M[i * n + j];
}

inline const double &at(const Matrix &M, int n, int i, int j) {
    return M[i * n + j];
}

Matrix random_dense_matrix(int n, double low = -1.0, double high = 1.0, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(low, high);
    Matrix M(n * n);
    for (int i = 0; i < n * n; ++i) {
        M[i] = dist(gen);
    }
    return M;
}

bool matrices_almost_equal(const Matrix &A, const Matrix &B, int n, double eps = 1e-9) {
    for (int i = 0; i < n * n; ++i) {
        if (std::fabs(A[i] - B[i]) > eps) {
            return false;
        }
    }
    return true;
}

double time_ms(const std::function<void()> &fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    return diff.count();
}

void matmul_baseline(const Matrix &A, const Matrix &B, Matrix &C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += at(A, n, i, k) * at(B, n, k, j);
            }
            at(C, n, i, j) = sum;
        }
    }
}

void transpose(const Matrix &B, Matrix &BT, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            at(BT, n, j, i) = at(B, n, i, j);
}

void matmul_transposed(const Matrix &A, const Matrix &B, Matrix &C, int n) {
    Matrix BT(n * n);
    transpose(B, BT, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += at(A, n, i, k) * at(BT, n, j, k);
            }
            at(C, n, i, j) = sum;
        }
    }
}

void matmul_blocked(const Matrix &A, const Matrix &B, Matrix &C, int n, int blockSize = 32) {
    std::fill(C.begin(), C.end(), 0.0);

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int kk = 0; kk < n; kk += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                int iMax = std::min(ii + blockSize, n);
                int kMax = std::min(kk + blockSize, n);
                int jMax = std::min(jj + blockSize, n);

                for (int i = ii; i < iMax; ++i) {
                    for (int k = kk; k < kMax; ++k) {
                        double a_ik = at(A, n, i, k);
                        for (int j = jj; j < jMax; ++j) {
                            at(C, n, i, j) += a_ik * at(B, n, k, j);
                        }
                    }
                }
            }
        }
    }
}

struct CSRMatrix {
    int n;
    std::vector<double> values;
    std::vector<int> col_index;
    std::vector<int> row_ptr;
};

CSRMatrix random_sparse_csr(int n, double density, unsigned seed = 42) {
    CSRMatrix A;
    A.n = n;
    A.row_ptr.resize(n + 1, 0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> valdist(-1.0, 1.0);

    for (int i = 0; i < n; ++i) {
        A.row_ptr[i] = static_cast<int>(A.values.size());
        for (int j = 0; j < n; ++j) {
            if (prob(gen) < density) {
                double v = valdist(gen);
                if (std::fabs(v) < 1e-12) {
                    continue;
                }
                A.values.push_back(v);
                A.col_index.push_back(j);
            }
        }
    }
    A.row_ptr[n] = static_cast<int>(A.values.size());
    return A;
}

Matrix csr_to_dense(const CSRMatrix &A) {
    int n = A.n;
    Matrix M(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];
        for (int idx = row_start; idx < row_end; ++idx) {
            int j = A.col_index[idx];
            M[i * n + j] = A.values[idx];
        }
    }
    return M;
}

void matmul_csr_dense(const CSRMatrix &A, const Matrix &B, Matrix &C) {
    int n = A.n;
    std::fill(C.begin(), C.end(), 0.0);

    for (int i = 0; i < n; ++i) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A.col_index[idx];
            double val = A.values[idx];

            const double *Brow = &B[col * n];
            double *Crow = &C[i * n];
            for (int j = 0; j < n; ++j) {
                Crow[j] += val * Brow[j];
            }
        }
    }
}

CSRMatrix loadMatrixMarketCSR(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open Matrix Market file: " + filename);
    }

    std::string line;

    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Unexpected EOF while reading header");
        }
    } while (!line.empty() && line[0] == '%');

    std::istringstream hdr(line);
    int rows, cols;
    long long nnz;
    if (!(hdr >> rows >> cols >> nnz)) {
        throw std::runtime_error("Failed to read matrix dimensions from header");
    }

    std::vector<std::tuple<int,int,double>> entries;
    entries.reserve(static_cast<size_t>(nnz));

    int i, j;
    double v;
    while (in >> i >> j >> v) {
        entries.emplace_back(i - 1, j - 1, v);
    }

    if (entries.size() != static_cast<size_t>(nnz)) {
        std::cerr << "Warning: expected nnz=" << nnz
                  << " but read " << entries.size() << " entries\n";
        nnz = static_cast<long long>(entries.size());
    }

    CSRMatrix A;
    A.n = rows;
    A.row_ptr.assign(rows + 1, 0);
    A.col_index.resize(nnz);
    A.values.resize(nnz);

    for (const auto &e : entries) {
        int r = std::get<0>(e);
        ++A.row_ptr[r + 1];
    }

    for (int r = 0; r < rows; ++r) {
        A.row_ptr[r + 1] += A.row_ptr[r];
    }

    std::vector<int> nextInRow(A.row_ptr.begin(), A.row_ptr.end());

    for (const auto &e : entries) {
        int r = std::get<0>(e);
        int c = std::get<1>(e);
        double value = std::get<2>(e);

        int pos = nextInRow[r]++;
        A.col_index[pos] = c;
        A.values[pos]    = value;
    }

    return A;
}

void csr_times_dense(const CSRMatrix &A,
                     const std::vector<double> &B,
                     std::vector<double> &C,
                     int K) {
    int n = A.n;
    std::fill(C.begin(), C.end(), 0.0);

    for (int i = 0; i < n; ++i) {
        int row_start = A.row_ptr[i];
        int row_end   = A.row_ptr[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int j = A.col_index[idx];
            double a = A.values[idx];

            const double *Bj = &B[j * K];
            double *Ci = &C[i * K];

            for (int k = 0; k < K; ++k) {
                Ci[k] += a * Bj[k];
            }
        }
    }
}

void benchmark_dense_variants(const std::vector<int> &sizes, double efficient_threshold_ms) {
    std::cout << "=== Dense multiplication: baseline vs transposed vs blocked ===\n";
    std::cout << std::fixed << std::setprecision(2);

    std::ofstream csv("dense_results.csv");
    csv << "N,method,time_ms,speedup_vs_baseline,correct,efficient,baseline_estimated\n";

    double last_baseline_time = std::numeric_limits<double>::quiet_NaN();
    int    last_baseline_n    = -1;

    for (int n : sizes) {
        std::cout << "\nMatrix size N = " << n << "\n";

        bool run_baseline = (n <= 2048);

        Matrix A = random_dense_matrix(n, -1.0, 1.0, 123);
        Matrix B = random_dense_matrix(n, -1.0, 1.0, 456);
        Matrix C_baseline(n * n);
        Matrix C_opt1(n * n);
        Matrix C_opt2(n * n);

        double t_base = std::numeric_limits<double>::quiet_NaN();
        double baseline_est = std::numeric_limits<double>::quiet_NaN();

        if (run_baseline) {
            matmul_baseline(A, B, C_baseline, n);

            t_base = time_ms([&]() {
                matmul_baseline(A, B, C_baseline, n);
            });

            last_baseline_time = t_base;
            last_baseline_n    = n;

            std::cout << "Baseline:   " << t_base << " ms\n";
        } else {
            if (last_baseline_n > 0 && std::isfinite(last_baseline_time)) {
                double factor = std::pow(double(n) / double(last_baseline_n), 3.0);
                baseline_est = last_baseline_time * factor;
                std::cout << "Baseline:   SKIPPED (estimated ~ " << baseline_est << " ms)\n";
            } else {
                std::cout << "Baseline:   SKIPPED (no estimate available)\n";
            }
        }

        double t_opt1 = time_ms([&]() {
            matmul_transposed(A, B, C_opt1, n);
        });

        double t_opt2 = time_ms([&]() {
            matmul_blocked(A, B, C_opt2, n, 32);
        });

        std::string corr1 = "SKIP";
        std::string corr2 = "SKIP";

        if (run_baseline) {
            bool ok1 = matrices_almost_equal(C_baseline, C_opt1, n);
            bool ok2 = matrices_almost_equal(C_baseline, C_opt2, n);
            corr1 = ok1 ? "YES" : "NO";
            corr2 = ok2 ? "YES" : "NO";
        }

        double base_for_speedup = run_baseline ? t_base : baseline_est;

        auto print_method = [&](const std::string &name,
                                double t,
                                const std::string &corr) {
            double speedup = std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(base_for_speedup) && t > 0.0) {
                speedup = base_for_speedup / t;
            }

            std::cout << name << ": " << std::setw(8) << t << " ms";
            if (std::isfinite(speedup)) {
                std::cout << "  | speedup"
                          << (run_baseline ? " = " : " (est) = ")
                          << speedup << "x";
            } else {
                std::cout << "  | speedup = NA";
            }
            std::cout << "  | correct=" << corr << "\n";

            bool efficient = (t <= efficient_threshold_ms);
            csv << n << "," << name << "," << t << ",";
            if (std::isfinite(speedup)) csv << speedup; else csv << "NA";
            csv << "," << corr << "," << (efficient ? "YES" : "NO") << ",";
            if (!run_baseline && std::isfinite(baseline_est)) {
                csv << "EST";
            } else if (run_baseline) {
                csv << "REAL";
            } else {
                csv << "NA";
            }
            csv << "\n";
        };

        print_method("OptTranspose", t_opt1, corr1);
        print_method("OptBlocked",   t_opt2, corr2);

        if (run_baseline) {
            bool eff_base = (t_base <= efficient_threshold_ms);
            csv << n << "," << "Baseline" << "," << t_base << ",";
            csv << 1.0 << "," << "YES" << "," << (eff_base ? "YES" : "NO") << ",REAL\n";
        } else if (std::isfinite(baseline_est)) {
            bool eff_base_est = (baseline_est <= efficient_threshold_ms);
            csv << n << "," << "Baseline_EST" << "," << baseline_est << ",";
            csv << "NA" << "," << "SKIP" << "," << (eff_base_est ? "YES" : "NO") << ",EST\n";
        }

        std::cout << "Efficient? (threshold " << efficient_threshold_ms << " ms):\n";
        if (run_baseline) {
            std::cout << "  Baseline: " << (t_base <= efficient_threshold_ms ? "YES" : "NO") << "\n";
        } else if (std::isfinite(baseline_est)) {
            std::cout << "  Baseline (est): "
                      << (baseline_est <= efficient_threshold_ms ? "YES" : "NO") << "\n";
        } else {
            std::cout << "  Baseline: UNKNOWN\n";
        }
        std::cout << "  Opt #1:   " << (t_opt1 <= efficient_threshold_ms ? "YES" : "NO") << "\n";
        std::cout << "  Opt #2:   " << (t_opt2 <= efficient_threshold_ms ? "YES" : "NO") << "\n";
    }

    std::cout << "\nDense results written to dense_results.csv\n";
}

void benchmark_sparse_vs_dense(int n, const std::vector<double> &densities, double efficient_threshold_ms) {
    std::cout << "\n=== Sparse (CSR) * dense vs dense baseline, N = " << n << " ===\n";
    std::cout << std::fixed << std::setprecision(2);

    std::ofstream csv("sparse_results.csv");
    csv << "N,density,nnz,sparsity,dense_time_ms,csr_time_ms,speedup,correct,efficient_dense,efficient_csr\n";

    Matrix B = random_dense_matrix(n, -1.0, 1.0, 789);
    Matrix C_dense(n * n);
    Matrix C_sparse(n * n);

    for (double density : densities) {
        std::cout << "\nDensity = " << (density * 100.0) << "% non-zero\n";

        CSRMatrix A_sparse = random_sparse_csr(n, density, 123);
        std::size_t nnz = A_sparse.values.size();
        double sparsity = 1.0 - (double)nnz / (double)(n * n);

        Matrix A_dense = csr_to_dense(A_sparse);

        double t_dense = time_ms([&]() {
            matmul_baseline(A_dense, B, C_dense, n);
        });

        double t_sparse = time_ms([&]() {
            matmul_csr_dense(A_sparse, B, C_sparse);
        });

        bool ok = matrices_almost_equal(C_dense, C_sparse, n);

        std::cout << "nnz = " << nnz << " / " << (long long)n * n
                  << "  (sparsity ~ " << (sparsity * 100.0) << "% zeros)\n";
        std::cout << "Dense baseline: " << t_dense << " ms\n";
        std::cout << "CSR * dense:    " << t_sparse << " ms  | speedup = " << (t_dense / t_sparse)
                  << "x  | correct=" << (ok ? "YES" : "NO") << "\n";

        bool eff_dense = (t_dense <= efficient_threshold_ms);
        bool eff_csr   = (t_sparse <= efficient_threshold_ms);

        std::cout << "Efficient? (threshold " << efficient_threshold_ms << " ms):\n";
        std::cout << "  Dense: " << (eff_dense ? "YES" : "NO") << "\n";
        std::cout << "  CSR:   " << (eff_csr ? "YES" : "NO") << "\n";

        csv << n << "," << density << "," << nnz << "," << sparsity << ","
            << t_dense << "," << t_sparse << "," << (t_dense / t_sparse) << ","
            << (ok ? "YES" : "NO") << ","
            << (eff_dense ? "YES" : "NO") << ","
            << (eff_csr ? "YES" : "NO") << "\n";
    }

    std::cout << "\nSparse results written to sparse_results.csv\n";
}

void fill_random(std::vector<double> &v) {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &x : v) x = dist(rng);
}

double milliseconds_since(const std::chrono::high_resolution_clock::time_point &start,
                          const std::chrono::high_resolution_clock::time_point &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void benchmark_mc2depi(const std::string &mtxPath,
                       int K,
                       int repeats = 3) {
    std::cout << "\n=== Real sparse matrix benchmark (mc2depi) ===\n";

    auto load_start = std::chrono::high_resolution_clock::now();
    CSRMatrix A = loadMatrixMarketCSR(mtxPath);
    auto load_end = std::chrono::high_resolution_clock::now();

    double load_ms = milliseconds_since(load_start, load_end);

    std::cout << "Loaded matrix from: " << mtxPath << "\n";
    std::cout << "Size: " << A.n << " x " << A.n
              << ", nnz = " << A.values.size() << "\n";

    double density = A.values.size() / (double(A.n) * double(A.n));
    std::cout << "Density ≈ " << density * 100.0 << "% ("
              << (100.0 - density * 100.0) << "% zeros)\n";
    std::cout << "Load time: " << load_ms << " ms\n";

    int n = A.n;
    std::vector<double> B(n * K);
    std::vector<double> C(n * K);

    fill_random(B);

    csr_times_dense(A, B, C, K);

    double best_ms = std::numeric_limits<double>::max();
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        csr_times_dense(A, B, C, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = milliseconds_since(t0, t1);
        best_ms = std::min(best_ms, ms);
    }

    double flops = 2.0 * double(A.values.size()) * double(K);
    double gflops = (flops / 1e9) / (best_ms / 1000.0);

    std::cout << "SpMM: A(" << n << "x" << n << ") * B(" << n << "x" << K << ")\n";
    std::cout << "Best time over " << repeats << " runs: " << best_ms << " ms\n";
    std::cout << "Approx. throughput: " << gflops << " GFLOP/s\n";
    std::cout << "Note: storing A as dense would require ~2.2 TB of RAM at this size,\n"
                 "so only a sparse (CSR) implementation is feasible.\n";

    std::ofstream csv("mc2depi_results.csv");
    csv << "n,nnz,density,load_time_ms,best_time_ms,gflops\n";
    csv << n << "," << A.values.size() << "," << density << ","
        << load_ms << "," << best_ms << "," << gflops << "\n";

    std::cout << "mc2depi results written to mc2depi_results.csv\n";
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    double efficient_threshold_ms = 120000.0;

    benchmark_dense_variants(sizes, efficient_threshold_ms);

    int sparseN = 1000;
    std::vector<double> densities = {0.01, 0.05, 0.1, 0.2, 0.5};
    benchmark_sparse_vs_dense(sparseN, densities, efficient_threshold_ms);

    try {
        benchmark_mc2depi("mc2depi.mtx", 16, 3);
    } catch (const std::exception &ex) {
        std::cerr << "mc2depi benchmark skipped: " << ex.what() << "\n";
    }

    return 0;
}
