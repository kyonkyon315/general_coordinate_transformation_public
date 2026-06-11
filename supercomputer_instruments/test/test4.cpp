//concept 練習
#include "../axis.h"
#include <concepts>

template<Axis_T T>
void dosomething(T x){
    auto p = T::num_blocks;
}

int main(){
    Axis<0,3,3,3> a(4);

    dosomething(a);
    //dosomething(b);//error
    return 0;
}