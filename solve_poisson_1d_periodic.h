#ifndef SOLVE_POISSON_1D_PERIODIC_H
#define SOLVE_POISSON_1D_PERIODIC_H

#include <vector>
#include <cmath>
#include <cassert>

// 物理定数（無次元化済み）
constexpr double poisson_coef = 1.0; // Derived<Plasma>::poisson_coef

// rho: 電荷密度（x方向グリッド数 = N）
// E: 電場（x方向グリッド数 = N+1, Yee格子用に境界を含む）
void solve_poisson_1d_periodic(const std::vector<double>& rho, std::vector<double>& E, double dx) {
    int N = rho.size();
    assert(E.size() == N + 1); // Yee格子用にセル境界も含む
    
    // 平均電荷を除去して周期境界条件対応
    double rho_avg = 0.0;
    for(int i=0;i<N;i++) rho_avg += rho[i];
    rho_avg /= N;

    // 電場の積分
    E[0] = 0.0; // 基準値（ポテンシャル自由度）
    for(int i=0;i<N;i++){
        E[i+1] = E[i] + dx * poisson_coef * (rho[i] - rho_avg);
    }
    
    // 平均を引いて、周期境界条件を調整
    double E_mean = 0.0;
    for(int i=0;i<=N;i++) E_mean += E[i];
    E_mean /= (N+1);

    for(int i=0;i<=N;i++) E[i] -= E_mean;
}


#endif //SOLVE_POISSON_1D_PERIODIC_H