#ifndef DISPERSION_RELATION_H
#define DISPERSION_RELATION_H

#include <cmath>
#include <limits>
#include "utils/constexpr_sqrt.h"
namespace DispersionRelation{
//--------------------------------------------------------------------
// k_hat_R(omega)
//--------------------------------------------------------------------
static constexpr double k_hat_R(double w,double wp_wc,double vth_c)
{
    double denom = w * (w - 1.0);

    // avoid singularity
    if (std::abs(denom) < 1e-14)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double n2 = 1.0 - (wp_wc * wp_wc) / denom;

    if (n2 < 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return (w * vth_c) * Utils::ConstExpr::sqrt(n2);
}

//--------------------------------------------------------------------
// solve omega from k using binary search
//
// solves:
//     k_hat_R(omega) = k_target
//
// search region:
//     omega > 1
//--------------------------------------------------------------------
static constexpr double omega_from_k(double k_target,double wp_wc,double vth_c,
                    double w_left  = 0.1,
                    double w_right = 0.99,
                    double tol     = 1e-10,
                    int max_iter   = 1000)
{
    for (int iter = 0; iter < max_iter; ++iter)
    {
        double w_mid = 0.5 * (w_left + w_right);

        double k_mid = k_hat_R(w_mid,wp_wc,vth_c);

        if (std::isnan(k_mid))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // convergence
        if (std::abs(k_mid - k_target) < tol)
        {
            return w_mid;
        }

        // binary search
        if (k_mid > k_target)
        {
            w_right = w_mid;
        }
        else
        {
            w_left = w_mid;
        }
    }

    return 0.5 * (w_left + w_right);
}
}//namespace DispersionRelation

#endif //DISPERSION_RELATION_H