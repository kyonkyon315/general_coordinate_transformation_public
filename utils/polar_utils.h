#ifndef POLAR_UTILS_H
#define POLAR_UTILS_H

#include <vector>
#include <cmath>
#include <vector>
#include <cmath>

class AdvectionAreaFlux {
private:
    const int N_max;
    const double A;
    const double dtheta;
    
    // 定数・係数の事前計算
    const double sqrt_A;
    const double inv_A_dtheta;
    const double cot_dtheta;
    const double I1_2;

    // ルックアップテーブル (インデックスは n または k に対応)
    std::vector<double> W_k;      // n-m > 1 用の幾何係数 W_k = 1/l^2 int_m^m+1 f_1^n (xi_2) dxi_2  この積分は、k=n-m にのみ依存します。2<= k <=N_max -1
    std::vector<double> N_factor; // 交点計算の分子
    std::vector<double> D_factor; // 交点計算の分母の第2項
    std::vector<double> V_n;      // n-m = 1 用の f2 積分係数
    std::vector<double> U_n;      // n-m = 1 用の cot(n*dtheta)
    std::vector<double> pow2_sin_n;      // pow2_sin_n[i] = sin^2(i dtheta)

    // インライン cotangent
    inline double cot(double x) const {
        return 1.0 / std::tan(x);
    }

public:
    AdvectionAreaFlux(int N_max_in, double A_in) 
        : N_max(N_max_in), 
          A(A_in), 
          dtheta(M_PI / N_max_in),
          sqrt_A(std::sqrt(A_in)),
          inv_A_dtheta(1.0 / (A_in * dtheta)),
          cot_dtheta(cot(dtheta)),
          I1_2(cot_dtheta - cot(2.0 * dtheta)),
          W_k(N_max_in + 1, NAN),
          N_factor(N_max_in + 1, 0.0),
          D_factor(N_max_in + 1, 0.0),
          V_n(N_max_in + 1, 0.0),
          U_n(N_max_in + 1, 0.0),
          pow2_sin_n(N_max_in + 1, 0.0)
    {
        // ルックアップテーブルの事前構築
        for (int i = 1; i <= N_max; ++i) {
            double theta_i = i * dtheta;
            double sin_theta_i = std::sin(theta_i);
            
            // n-m = 1 用のテーブル (i を n として扱う)
            N_factor[i] = sin_theta_i;
            D_factor[i] = std::cos(theta_i);
            V_n[i] = sin_theta_i * sin_theta_i / dtheta;
            U_n[i] = cot(theta_i);

            // n-m > 1 用のテーブル (i を k(=n-m) として扱う)
            if (i >= 2 && i < N_max) {
                //W_k[i] = cot((i-1)*dtheta) - cot(i*dtheta);
                W_k[i] = sin(dtheta)/sin((double)i*dtheta)/sin((double)(i-1)*dtheta);
            }
        }
        W_k[N_max]=0.;

        for(int i=0;i<=N_max; ++i){
            pow2_sin_n[i] = sin((double)i * dtheta) * sin((double)i * dtheta);
        }
    }

    // セル面積 S_{m,n} の計算 (ホットスループット用)
    inline double compute_S(int m, int n, double l) const {
        const int k = n - m;
        const double l_sq_inv_A_dtheta = l * l * inv_A_dtheta;

        if (k > 1) {
            // 解析的に求めた定数テーブルとの積のみで O(1) 計算
            //std::cout<<"k="<<k<<" "<<W_k[k]<<" "<<W_k[k+1]<<" w\n";
            return l_sq_inv_A_dtheta * (W_k[k]*pow2_sin_n[n] - W_k[k+1]*pow2_sin_n[n+1]);
        } 
        else if (k == 1) {
            const double l_over_sqrt_A = l / sqrt_A;
            
            // 交点 xi_cross の計算 (atan は値域的に安全)
            //double xi_c = std::atan(N_factor[n] / (l_over_sqrt_A + D_factor[n])) / dtheta;
            double xi_c = (double)n - std::asin(l_over_sqrt_A*std::sin((double)n * dtheta)) / dtheta;

            //std::cout<<"xis_c:"<<xi_c;
            //std::cout<<"L:"<< l*l/A<<" ";
            
            // 丸め誤差等による区間逸脱を防ぐためクランプ
            //xi_c = std::clamp(xi_c, static_cast<double>(m), static_cast<double>(n));

            // \int_m^{xi_c} f_1 d\xi_2
            const double term1 = l_sq_inv_A_dtheta * pow2_sin_n[n] * (cot((n-xi_c) * dtheta) - cot(k * dtheta));
            
            // \int_{xi_c}^{m+1} f_2 d\xi_2
            //const double term2 = V_n[n] * (cot(xi_c * dtheta) - cot((double)(m+1) * dtheta));
            const double term2 = m+1 - xi_c;
            
            // - \int_m^{m+1} f_1^{m+2} d\xi_2 (事前計算済)
            const double term3 = l_sq_inv_A_dtheta * W_k[k + 1]*pow2_sin_n[n+1];
            //std::cout<<"sum:"<<term1 + term2 - term3<<"\n";

            return term1 + term2 - term3;
        }

        return 0.0; // k <= 0 は移流方向外
    }
};
#endif //POLAR_UTILS_H
