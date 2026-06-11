#ifndef SINGULARITY_PROCESSOR_H
#define SINGULARITY_PROCESSOR_H
#include <cmath>
#include "integrator_fast.h"
//#include "integrator.h"

//#define DEBUG_SINGULARITY
#ifdef DEBUG_SINGULARITY
#include <iostream>
#endif
using Value = double;

namespace PolarCoordinates{

class SingularityProcessor{
private:
    const int N_max;
    const Value d_theta;
    int k;
    Integrator integrator;

    Value calc_area_small_phi(int m,int n)const{
#ifdef DEBUG_SINGULARITY
        if(n<=m || m<0 || n>=2*N_max){
            std::cout<<"error: [l<0. || beta<0. || beta>=1.||n<=m || m<0 || n>=2*N_max] in singularity_processor.h\n";
            std::cout<<"("<<n<<","<<m<<")\n";
            throw 1;
        }
#endif

        if(n<= N_max-1){
            if(m==0){
                //std::cout<<"integrator.calc(l, n, beta, beta, 1):"<<integrator.calc(l, n, beta, beta, 1)<<"\n";
                //std::cout<<"integrator.calc(l, n, beta, beta, 1):"<<integrator.calc(l, n+1, beta, beta, 1)<<"\n";
                return integrator.calc_beta_1(n) - integrator.calc_beta_1(n+1);
            }
            else if(m<= n-1){
                return integrator.calc(n, m, m+1) - integrator.calc(n+1, m, m+1);
            }
#ifdef DEBUG_SINGULARITY
            else{
                std::cout<<"bad argument: type 1 (m,n):("<<m<<","<<n<<") .\n";
                throw 1;
            }
#endif
        }
        else if(n==N_max){
            if(m==0){
                return integrator.calc_beta_1(n) - integrator.calc_0_beta(n+1);
            }
            else if(m<=n-1){
                return integrator.calc(n, m, m+1);
            }
#ifdef DEBUG_SINGULARITY
            else{
                std::cout<<"bad argument: type 2 (m,n):("<<m<<","<<n<<") .\n";
                throw 1;
            }
#endif
        }
        else {
            if (m == 0) {    
                return integrator.calc_0_beta(n+1) - integrator.calc_0_beta(n);
            }
            else if(m<=N_max-1){
                return 0.;
            }
            else if(m==N_max){
                return - integrator.calc(m+1, n, n+1);
            }
            else if(m<= n-1){
                return integrator.calc(m,n, n+1)- integrator.calc(m+1, n, n+1);
            }
#ifdef DEBUG_SINGULARITY
            else{
                std::cout<<"bad argument: type 3 (m,n):("<<m<<","<<n<<") .\n";
                throw 1;
            }
#endif
        }
        return 0.;
    }
public:

    SingularityProcessor(int N_max,double A):
        N_max(N_max),
        d_theta(M_PI/(Value)N_max),
        integrator(N_max, A)
    {}
    void set_l_and_phi(const Value l_, const Value phi) {
        k = std::floor(phi/d_theta);
        if(phi<(Value)k*d_theta)k--;
        if(phi>=(Value)(k+1)*d_theta)k++;
#ifdef DEBUG_SINGULARITY
        if(!((Value)k*d_theta<=phi && phi<(Value)(k+1)*d_theta)){
            std::cout<<"k error\n";
            throw 1;
        }
#endif
        const Value beta = (phi - d_theta*(Value)k)/d_theta;
        integrator.set_l_and_beta(l_, beta);
#ifdef DEBUG_SINGULARITY
        if(l_<0. || beta<0. || beta>=1.){
            std::cout<<"error: [l<0. || beta<0. || beta>=1.] in singularity_processor.h\n";
            std::cout<<"("<<l_<<","<<beta<<")\n";
            throw 1;
        }
#endif
    }

    Value calc_area(const int m,const int n)const{
#ifdef DEBUG_SINGULARITY
        if(n<=m || m<0 || n>=2*N_max){
            std::cout<<"error: [l<0. || beta<0. || beta>=1.||n<=m || m<0 || n>=2*N_max] in singularity_processor.h\n";
            throw 1;
        }
#endif
        if(m>=k){
            return calc_area_small_phi(m-k, n-k);
        }
        else if(n>=k){
            return -calc_area_small_phi(n-k, m-k+2*N_max);
        }
        else{
            return calc_area_small_phi(m-k+2*N_max, n-k+2*N_max);
        }
    }
};
}//namespace PolarCoordinates

#endif //SINGULARITY_PROCESSOR_H