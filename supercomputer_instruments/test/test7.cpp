#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

const int N = 1000000;
const int STEPS = 100;

// 重さを調整：SIMDが効きやすい演算セット
inline double hoge(int i) {
    double x = (double)i * 0.01;
    // 平方根や除算は重いので差が出やすい
    return std::sqrt(x + 1.0) + std::sin(x) * x;
}

int main() {
    // データ準備
    std::vector<double> a_origin(N, 0.0);
    std::vector<double> a_A = a_origin;
    std::vector<double> a_B = a_origin;

    std::cout << "Elements: " << N << ", Steps: " << STEPS << std::endl;

    // --- Method A (Scatter) ---
    auto start_A = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < STEPS; ++s) {
        // Method A: 従来の書き方
        // a[i-1] と a[i] への書き込みが依存し合う
        for (int i = 1; i < N; ++i) {
            double x = hoge(i);
            a_A[i - 1] -= x;
            a_A[i]     += x;
        }
    }
    auto end_A = std::chrono::high_resolution_clock::now();

    // --- Method B (Flux/Gather) ---
    // ★重要: メモリ確保はループの外でやる！
    // これを中に書くと malloc/free の時間ばかり計測することになります
    std::vector<double> flux(N);

    auto start_B = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < STEPS; ++s) {
        
        // 1. Flux一括計算 (SIMD OK)
        #pragma omp simd
        for (int i = 0; i < N; ++i) {
            flux[i] = hoge(i);
        }

        // 2. 差分更新 (SIMD OK)
        // Method Aのロジックと完全に一致させるための境界処理
        
        // 端の処理（SIMDループから外す）
        a_B[0] -= flux[1];      // Aでは i=1 のとき a[0] -= hoge(1)
        a_B[N-1] += flux[N-1];  // Aでは i=N-1 のとき a[N-1] += hoge(N-1)

        // 中身の処理
        // Aでは a[k] は、i=k のとき +=hoge(k)、i=k+1 のとき -=hoge(k+1) される
        // つまり a[k] += flux[k] - flux[k+1]
        // 範囲は k=1 から k=N-2 まで
        #pragma omp simd
        for (int k = 1; k < N - 1; ++k) {
            a_B[k] += flux[k] - flux[k+1];
        }
    }
    auto end_B = std::chrono::high_resolution_clock::now();

    // 結果出力
    double time_A = std::chrono::duration<double, std::milli>(end_A - start_A).count();
    double time_B = std::chrono::duration<double, std::milli>(end_B - start_B).count();

    std::cout << "Method A: " << time_A << " ms" << std::endl;
    std::cout << "Method B: " << time_B << " ms" << std::endl;
    std::cout << "Speedup: " << time_A / time_B << "x" << std::endl;

    // Diff check
    double diff = 0.0;
    for(int i=0; i<N; ++i) diff += std::abs(a_A[i] - a_B[i]);
    std::cout << "Diff check: " << diff << std::endl;
    
    // 最初の数個の値を表示して確認
    if(diff > 1e-9) {
        std::cout << "Example mismatch:" << std::endl;
        for(int i=0; i<5; ++i) 
            printf("i=%d: A=%f, B=%f\n", i, a_A[i], a_B[i]);
    }

    return 0;
}