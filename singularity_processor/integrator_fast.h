#ifndef INTEGRATOR_H
#define INTEGRATOR_H

//#define DEBUG_INTEGRATOR

#ifdef DEBUG_INTEGRATOR
#include <iostream>
#endif
#include <cmath>
#include <vector>

using Value = double;

namespace PolarCoordinates {

class Integrator {
private:
    const int N_max;
    const Value d_theta;
    const Value A;
    Value l;
    Value beta;

    // --- 事前計算用キャッシュテーブル ---
    // 1. コンストラクタで一度だけ計算（l, beta に依存しない）
    // 整数 id に対する -1.0 / (d_theta * tan((id - i) * d_theta)) を格納
    // インデックスの範囲は (id - i) + N_max としてみなす
    std::vector<Value> inv_tan_table;

    // 2. set_l_and_beta で計算（i に依存する値をキャッシュ）
    std::vector<Value> x2_cache;          // term1 (x^2) のキャッシュ
    std::vector<Value> asin_x_div_dtheta; // asin(x) / d_theta のキャッシュ
    std::vector<Value> int_xi_c_pure;     // int_one_side(i, xi_c) の基本形のキャッシュ
    std::vector<Value> int_beta_cache;    // int_one_side(i, beta) のキャッシュ

public:
    Integrator(int N_max, double A) :
        N_max(N_max),
        d_theta(M_PI / (Value)N_max),
        A(A),
        l(0),
        beta(0)
    {
        // 整数用 tan テーブルの事前計算 (範囲: id - i が -N_max から 2*N_max まで)
        inv_tan_table.resize(3 * N_max + 1);
        for (int k = -N_max; k <= 2 * N_max; ++k) {
            if (k % N_max == 0) {
                inv_tan_table[k + N_max] = 0.0; // ガードされるためアクセスされない
            } else {
                inv_tan_table[k + N_max] = -1.0 / (d_theta * std::tan(k * d_theta));
            }
        }

        // キャッシュ領域の確保
        x2_cache.resize(N_max + 1);
        asin_x_div_dtheta.resize(N_max + 1);
        int_xi_c_pure.resize(N_max + 1);
        int_beta_cache.resize(N_max + 1);
    }

    void set_l_and_beta(const Value l_, const Value beta_) {
        l = l_;
        beta = beta_;

        const Value sqrt_A = std::sqrt(A);

        // i = 0 から N_max までの超越関数をここで一挙に計算
        for (int i = 0; i <= N_max; ++i) {
            const Value sin_up = std::sin(((Value)i - beta) * d_theta);
            const Value x = std::abs(l / sqrt_A * sin_up);

            x2_cache[i] = x * x; // 元の term1 は数学的に x^2 と等価になります
            asin_x_div_dtheta[i] = std::asin(x) / d_theta;
            
            // 数学的な変形により、int_one_side(i, xi_c) から tan と asin が消去できます
            int_xi_c_pure[i] = x * std::sqrt(1.0 - x * x) / d_theta;

            // int_one_side(i, beta) のキャッシュ (beta < 1e-2 のケアも含む)
            if (i == 0) {
                if (beta < 1e-2) {
                    int_beta_cache[0] = -l * l * beta / A;
                } else {
                    int_beta_cache[0] = x2_cache[0] * (-1.0 / d_theta / std::tan(beta * d_theta));
                }
            } else {
                int_beta_cache[i] = x2_cache[i] * (-1.0 / d_theta / std::tan((beta - (Value)i) * d_theta));
            }
        }
    }

    Value calc(int i, int left, int right) const {
#ifdef DEBUG_INTEGRATOR
        if (left == 0 || right - left != 1) {
            std::cout << "left==0 || right-left!=1\n";
            throw 1;
        }
        if (i < 0 || right - left != 1 || left < 0 || right > 2 * N_max) {
            std::cout << "error: G_i_j1\n";
            throw 1;
        }
#endif
        i %= N_max;
        if (i == 0) i = N_max;

        if (left <= N_max - 1) {
#ifdef DEBUG_INTEGRATOR
            if (1 <= i && i <= left) {

                std::cout << "error: G_i_j2\n";
                throw 1;
            }
#endif
            if (left == i - 1) {
                if (i == N_max && beta < 1e-9) return 0.;
                
                const Value xi_c = (Value)i - asin_x_div_dtheta[i];
                // int_one_side(i, xi_c) => int_xi_c_pure[i]
                // int_one_side(i, left) => x2_cache[i] * inv_tan_table[...]
                return (Value)right - xi_c + int_xi_c_pure[i] - (x2_cache[i] * inv_tan_table[left - i + N_max]);
            } else {
                return x2_cache[i] * (inv_tan_table[right - i + N_max] - inv_tan_table[left - i + N_max]);
            }
        } 
#ifdef DEBUG_INTEGRATOR
        else if (left == N_max) {
            std::cout << "error: G_i_j3\n";
            throw 1;
        } 
#endif
        else {
            if (left == N_max + i) {
                const Value xi_c = (Value)(N_max + i) + asin_x_div_dtheta[i];
                // このケースの int_one_side(i, xi_c) は数学的に -int_xi_c_pure[i] になります
                return xi_c - (Value)left + (x2_cache[i] * inv_tan_table[right - i + N_max]) + int_xi_c_pure[i];
            } else {
                return x2_cache[i] * (inv_tan_table[right - i + N_max] - inv_tan_table[left - i + N_max]);
            }
        }
    }

    Value calc_beta_1(int i) const {
        i %= N_max;
        if (i == 1) {
            const Value xi_c = 1.0 - asin_x_div_dtheta[1];
            return 1.0 - xi_c + int_xi_c_pure[1] - int_beta_cache[1];
        } else {
            return (x2_cache[i] * inv_tan_table[1 - i + N_max]) - int_beta_cache[i];
        }
    }

    Value calc_0_beta(int i) const {
        i %= N_max;
        if (i == 0) {
            const Value xi_c = asin_x_div_dtheta[0];
            const Value term1 = xi_c;
            const Value term2 = int_beta_cache[0];
            // 元の int_one_side(0, xi_c) は数学的に -int_xi_c_pure[0] と等価です
            const Value term3 = (beta < 1e-2 ? -l * beta / std::sqrt(A) : -int_xi_c_pure[0]);
            return term1 + term2 - term3;
        } else {
            return int_beta_cache[i] - (x2_cache[i] * inv_tan_table[0 - i + N_max]);
        }
    }
};

} // namespace PolarCoordinates

#endif // INTEGRATOR_H

/*元のコード

class Integrator{
private:
    const int N_max;
    const Value d_theta;
    const Value A;
    Value l;
    Value beta;

    Value int_one_side(int i, int id)const{
        // TODO テーブルにして高速化
        return int_one_side(i, (Value)id);
    }
    Value int_one_side(int i, Value id)const{

        const Value sin_up = sin(((Value)i - beta)*d_theta);
        const Value term1 = l*l *sin_up* sin_up / A;
        const Value term2 = - 1./d_theta /tan((id-(Value)i)*d_theta);
        return term1 * term2;
    }
public:

    Integrator(int N_max,double A):
        N_max(N_max),
        d_theta(M_PI/(Value)N_max),
        A(A)
    {

    }

    Value calc(int i, int left, int right)const{
        if(left==0 || right-left!=1){
            std::cout<<"left==0 || right-left!=1\n";
            throw 1;
        }
        //std::cout<<"(Value l"<<l<<", int i"<<i<<", Value beta"<<beta<<", int left"<<left<<", int right"<<right<<")\n";
        if(i<0 || right-left != 1 ||left<0 || right>2*N_max){
            std::cout<<"error: G_i_j1\n";
            throw 1;
        }
        i%= N_max;
        if(i==0)i=N_max;
        if(left<=N_max-1){
            if(1<=i && i<=left){
                std::cout<<"error: G_i_j2\n";
                throw 1;
            }
            else{
                if(left == i-1){
                    if(i==N_max && beta<1e-9)return 0.;
                    Value xi_c = (Value)i-1./d_theta *std::asin(std::abs(l/sqrt(A) *sin(((Value)i-beta)*d_theta)));
                    return (Value)right-xi_c +int_one_side(i,xi_c)-int_one_side(i,left);
                }
                else{
                    return int_one_side(i,right)-int_one_side(i,left);
                }
            }
        }
        else if(left==N_max){
            std::cout<<"error: G_i_j3\n";
            throw 1;
        }
        else{
            if(left == N_max+i){
                Value xi_c = (Value)(N_max+i)+1./d_theta *std::asin(std::abs(l/sqrt(A) *sin(((Value)i-beta)*d_theta)));
                return xi_c-(Value)left +int_one_side(i,right)-int_one_side(i,xi_c);
            }
            else{
                return int_one_side(i,right)-int_one_side(i,left);
            }
        }
    }

    void set_l_and_beta(const Value l_,const Value beta_){
        l=l_;
        beta = beta_;
    }

    Value calc_beta_1(int i)const{
        i%=N_max;
        if(i==1){
            Value xi_c = (Value)i-1./d_theta *std::asin(std::abs(l/sqrt(A) *sin(((Value)i-beta)*d_theta)));
            return 1.-xi_c +int_one_side(i,xi_c)-int_one_side(i,beta);
        }
        else{
            return int_one_side(i,1.)-int_one_side(i,beta);
        }
    } 
    Value calc_0_beta(int i)const{
        
        i%=N_max;
        if(i==0){
            Value xi_c = (Value)i+1./d_theta *std::asin(std::abs(l/sqrt(A) *sin(((Value)i-beta)*d_theta)));
            //return xi_c +int_one_side(l,i,beta,beta)-int_one_side(l,i,beta,xi_c);
            const Value term1 = xi_c;
            const Value term2 = (beta<1e-2 ? -l*l*beta/A :int_one_side(i,beta));
            const Value term3 = (beta<1e-2 ? -l*beta/sqrt(A) : int_one_side(i,xi_c));
            return term1 + term2 - term3;
        }
        else{
            return int_one_side(i,beta)-int_one_side(i,0.);
        }  
    } 
};

}//namespace PolarCoordinates


*/