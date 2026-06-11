#ifndef CONSTEXPR_SQRT_H
#define CONSTEXPR_SQRT_H
#include <limits>

namespace Utils{
namespace ConstExpr{
    namespace Tools{
        constexpr double sqrt_helper(double s, double curr, double prev, int iter) {
            return (curr == prev || iter > 100) // 100回回れば十分すぎる
                ? curr
                : sqrt_helper(s, 0.5 * (curr + s / curr), curr, iter + 1);
        }
    }

    constexpr double sqrt(double s) {
    if (s < 0) return std::numeric_limits<double>::quiet_NaN();
        if (s == 0) return 0;
        
        // 初期の推測値として s を使用
        return Tools::sqrt_helper(s, s,-1., 0);
    }
}
}

#endif //CONSTEXPR_SQRT_H