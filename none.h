#ifndef NONE_H
#define NONE_H
#include <array>
class None_current{
public:
    static constexpr int get_dimension(){
        return 0;
    }
    static constexpr std::array<int,0> shape={};

};
#endif //NONE_H