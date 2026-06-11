#ifndef INDEPENDENT_H
#define INDEPENDENT_H
// --- (微分ゼロ) を表すクラス ---
using Value = double;
class Independent
{
public:
    Independent(){}
    // 常に 0.0 を返す
    //static constexpr Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp){
    template<typename... Ints>
    static constexpr Value at(Ints... calc_axes){
        return 0.0;
    }
};
#endif //INDEPENDENT_H
