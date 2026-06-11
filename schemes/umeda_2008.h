#ifndef UMEDA_2008_H
#define UMEDA_2008_H
#include <utility>
using Value = double;

/*

[Takayuki Umeda et.al 2008, "A conservative and non-oscillatory scheme for Vlasov code simulations"]

df/dt + v df/dx = 0
を解くルーチンをここに書く
nyu = - v Δt/Δx
とすると、

f(i,t+Δt) = f(i,t) + U_(i-1/2)(nyu) - U_(i+1/2)(nyu)

ただし、
U_(i+1/2)(nyu) = nyu f(i) + nyu(1-nyu)(2-nyu)L_i_positive/6
                            + nyu(1-nyu)(1+nyu)L_i_negative/6

L_i_positive =  min[2(f_i - f_min), (f_i+1 - f_i)] if f_i+1 >= f_i
                max[2(f_i - f_max), (f_i+1 - f_i)] else

L_i_negative =  min[2(f_max - f_i), (f_i - f_i-1)] if f_i >= f_i-1
                max[2(f_min - f_i), (f_i - f_i-1)] else

f_max = max[f_max1, f_max2]
f_min = min[f_min1, f_min2]

f_max1 = max[max[f_i-1, f_i], min[2f_i-1 - f_i-2, 2f_i - f_i+1]]
f_max2 = max[max[f_i+1, f_i], min[2f_i+1 - f_i+2, 2f_i - f_i-1]]
f_min1 = min[min[f_i-1, f_i], max[2f_i-1 - f_i-2, 2f_i - f_i+1]]
f_min2 = min[min[f_i+1, f_i], max[2f_i+1 - f_i+2, 2f_i - f_i-1]]

*/

#include <algorithm> // for std::min/std::max

using Value = double;

class Umeda2008{
private:
    // calc_flux_rightward は「中心 f0 を用いる (fm2,fm1,f0,fp1,fp2)」で
    // nyu >= 0 (左→右 情報) を想定した計算（先に示したものと同じ）
    static inline Value calc_flux_rightward(
        Value fm2, Value fm1, Value f0, Value fp1, Value fp2,
        Value nyu
    ){
        Value f_max1 = std::max(std::max(fm1, f0),
                        std::min(2.0*fm1 - fm2, 2.0*f0 - fp1));
        Value f_max2 = std::max(std::max(fp1, f0),
                        std::min(2.0*fp1 - fp2, 2.0*f0 - fm1));
        Value f_min1 = std::min(std::min(fm1, f0),
                        std::max(2.0*fm1 - fm2, 2.0*f0 - fp1));
        Value f_min2 = std::min(std::min(fp1, f0),
                        std::max(2.0*fp1 - fp2, 2.0*f0 - fm1));

        Value f_max = std::max(f_max1, f_max2);
        Value f_min = std::max(0.,std::min(f_min1, f_min2));

        Value L_pos;
        if (fp1 >= f0)
            L_pos = std::min(4.0*(f0 - f_min), (fp1 - f0));
        else
            L_pos = std::max(4.0*(f0 - f_max), (fp1 - f0));

        Value L_neg;
        if (f0 >= fm1)
            L_neg = std::min(3.0*(f_max - f0), (f0 - fm1));
        else
            L_neg = std::max(3.0*(f_min - f0), (f0 - fm1));

        Value U = (f0
                + (1.0 - nyu)*(2.0 - nyu)*L_pos/6.
                + (1.0 - nyu)*(1.0 + nyu)*L_neg/6.)*nyu;
        //Value U = nyu*fp1+nyu*(1-nyu)*(2-nyu)*(fp1-f0)/6.
        //                +nyu*(1-nyu)*(1+nyu)*(f0-fm1)/6.;//v>0のためのやつ
        return U;
    }

public:
    /********************************************************************************************
     * Limiterとして、関数calc_U_ip_half　は必須である。最後の引数がnyuであり、それ以外の引数が近接グリッドの
     * 分布関数値である。 
     * U[i+1/2]を出力する
     * f[i]の増加分を計算するために入力として受け取る分布関数がF[i-x]～F[i+y]のx+y+1個である場合、
     * ユーザーはused_id_left = -x,used_id_right = yとして定義しなければならない。
     *********************************************************************************************/
    static const int used_id_left  = -3;
    static const int used_id_right =  3;

    void calc_U(
        Value f_im3, Value f_im2, Value f_im1,
        Value f_i,
        Value f_ip1, Value f_ip2, Value f_ip3,
        Value nyu_m_half, Value nyu_p_half, Value& Um, Value& Up
    )const{
        nyu_p_half = -nyu_p_half;//なぜかnyuを反転しないと逆になってしまう。
        nyu_m_half = -nyu_m_half;//なぜかnyuを反転しないと逆になってしまう。
        Value U_ip_half;
        Value U_im_half;
        if (nyu_m_half >= 0.0){
            U_im_half = calc_flux_rightward(
                /*fm2*/ f_im3,  // i-2
                /*fm1*/ f_im2,  // i-1
                /*f0*/  f_im1,    // i
                /*fp1*/ f_i,  // i+1
                /*fp2*/ f_ip1,  // i+2
                nyu_m_half // nyu は既に正
            );
        }
        else{
            U_im_half = - calc_flux_rightward(
                /*fm2*/ f_ip2,
                /*fm1*/ f_ip1,
                /*f0*/  f_i,
                /*fp1*/ f_im1,
                /*fp2*/ f_im2,
                -nyu_m_half
            );
        }

        if (nyu_p_half >= 0.0){
            U_ip_half = calc_flux_rightward(
                /*fm2*/ f_im2,  
                /*fm1*/ f_im1,    
                /*f0*/  f_i,  
                /*fp1*/ f_ip1,  
                /*fp2*/ f_ip2,  
                nyu_p_half // nyu は既に正
            );
        }
        else{
            U_ip_half = - calc_flux_rightward(
                /*fm2*/ f_ip3,
                /*fm1*/ f_ip2,
                /*f0*/  f_ip1,
                /*fp1*/ f_i,
                /*fp2*/ f_im1,
                -nyu_p_half
            );
        }
        Um = U_im_half;
        Up = U_ip_half;
    }
};
#endif //UMEDA_2008_H
