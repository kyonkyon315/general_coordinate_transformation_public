

#include <random>
#include <iostream>

int main(){
    for(int i=0;i<3;i++){
        std::mt19937 rng(12345 );
        std::uniform_real_distribution<double> uni(-1.0,1.0);
        std::cout<<uni(rng)<<"\n";
    }
}

