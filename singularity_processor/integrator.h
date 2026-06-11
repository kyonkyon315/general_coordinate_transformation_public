#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include <iostream>
#include <cmath>
using Value = double;
namespace PolarCoordinates{

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

#endif //INTEGRATOR_H
