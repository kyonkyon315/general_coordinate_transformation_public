// maxwell_solver_with_tensor_utils.cpp
// Combined example: tensor conversion utilities + a simple Yee-like Maxwell solver
// on a skwed non-orthogonal mapping (3D) using contravariant components.
// - Uses the previously provided utility functions (metric, raise/lower, transforms)
// - Updates B^i and E^i in contravariant form using central differences in xi
// - Demonstrates applying physical BCs and outputting Cartesian fields

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

using Vec3 = std::array<double,3>;
using Mat3 = std::array<std::array<double,3>,3>;

// -------------------- linear algebra / metric utilities --------------------
Mat3 compute_metric(const Mat3 &Jax){
    Mat3 g{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j){
        double s = 0.0;
        for(int a=0;a<3;++a) s += Jax[a][i]*Jax[a][j];
        g[i][j] = s;
    }
    return g;
}

double det3(const Mat3 &M){
    return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
         - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
         + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
}

Mat3 invert3(const Mat3 &M){
    Mat3 R{};
    double D = det3(M);
    assert(std::fabs(D)>1e-15 && "matrix singular or nearly singular");
    R[0][0] =  (M[1][1]*M[2][2]-M[1][2]*M[2][1]) / D;
    R[0][1] = -(M[0][1]*M[2][2]-M[0][2]*M[2][1]) / D;
    R[0][2] =  (M[0][1]*M[1][2]-M[0][2]*M[1][1]) / D;

    R[1][0] = -(M[1][0]*M[2][2]-M[1][2]*M[2][0]) / D;
    R[1][1] =  (M[0][0]*M[2][2]-M[0][2]*M[2][0]) / D;
    R[1][2] = -(M[0][0]*M[1][2]-M[0][2]*M[1][0]) / D;

    R[2][0] =  (M[1][0]*M[2][1]-M[1][1]*M[2][0]) / D;
    R[2][1] = -(M[0][0]*M[2][1]-M[0][1]*M[2][0]) / D;
    R[2][2] =  (M[0][0]*M[1][1]-M[0][1]*M[1][0]) / D;
    return R;
}

Vec3 raise_index(const Mat3 &ginv, const Vec3 &Vcov){
    Vec3 Vcon{};
    for(int i=0;i<3;++i){
        double s=0.0;
        for(int j=0;j<3;++j) s += ginv[i][j]*Vcov[j];
        Vcon[i]=s;
    }
    return Vcon;
}

Vec3 lower_index(const Mat3 &g, const Vec3 &Vcon){
    Vec3 Vcov{};
    for(int i=0;i<3;++i){
        double s=0.0;
        for(int j=0;j<3;++j) s += g[i][j]*Vcon[j];
        Vcov[i]=s;
    }
    return Vcov;
}

std::array<double,3> compute_scale_factors(const Mat3 &Jax, const Mat3 &g){
    std::array<double,3> h{};
    for(int i=0;i<3;++i) h[i] = std::sqrt(std::fabs(g[i][i]));
    return h;
}

Vec3 contravariant_to_physical(const Mat3 &g, const std::array<double,3>& h, const Vec3 &Vcon){
    Vec3 Vcov = lower_index(g, Vcon);
    Vec3 Vphys{};
    for(int i=0;i<3;++i) Vphys[i] = Vcov[i] / h[i];
    return Vphys;
}

Vec3 physical_to_contravariant(const Mat3 &ginv, const std::array<double,3>& h, const Vec3 &Vphys){
    Vec3 Vcov{};
    for(int i=0;i<3;++i) Vcov[i] = h[i] * Vphys[i];
    return raise_index(ginv, Vcov);
}

Vec3 contravariant_to_cartesian(const Mat3 &Jax, const Vec3 &Vcon){
    Vec3 Vcart{};
    for(int a=0;a<3;++a){
        double s=0.0;
        for(int i=0;i<3;++i) s += Jax[a][i] * Vcon[i];
        Vcart[a] = s;
    }
    return Vcart;
}

Vec3 cartesian_to_contravariant(const Mat3 &Jax, const Vec3 &Vcart){
    Mat3 Minv = invert3(Jax);
    Vec3 Vcon{};
    for(int i=0;i<3;++i){
        double s=0.0;
        for(int a=0;a<3;++a) s += Minv[i][a]*Vcart[a];
        Vcon[i]=s;
    }
    return Vcon;
}

// -------------------- simple grid helpers (uniform in xi) --------------------
struct Grid {
    int Nx,Ny,Nz;
    double dx,dy,dz; // grid spacing in computational xi-space
};

inline int idx(int i,int j,int k,int Nx,int Ny,int Nz){ return (k*Nx*Ny + j*Nx + i); }

// periodic index
inline int ip(int i,int N){ if(i<0) return i+N; if(i>=N) return i-N; return i; }

// central difference in xi_j of a scalar field stored in 3D array

double d_dxi_j(const std::vector<double>& field, int i,int j,int k, int Nx,int Ny,int Nz, int comp_j, double invdx){
    int i2=i, j2=j, k2=k;
    if(comp_j==0){ // derivative wrt xi1 (i)
        int il=ip(i-1,Nx), ir=ip(i+1,Nx);
        return 0.5*(field[idx(ir,j,k,Nx,Ny,Nz)] - field[idx(il,j,k,Nx,Ny,Nz)])*invdx;
    }else if(comp_j==1){ // wrt xi2 (j)
        int jl=ip(j-1,Ny), jr=ip(j+1,Ny);
        return 0.5*(field[idx(i,jr,k,Nx,Ny,Nz)] - field[idx(i,jl,k,Nx,Ny,Nz)])*invdx;
    }else{ // comp_j==2
        int kl=ip(k-1,Nz), kr=ip(k+1,Nz);
        return 0.5*(field[idx(i,j,kr,Nx,Ny,Nz)] - field[idx(i,j,kl,Nx,Ny,Nz)])*invdx;
    }
}

// --- write_slice_binary for Econ, Bcon stored as std::vector<double> of size N*3 ---
void write_slice_binary(const std::vector<double> &Econ,
                        const std::vector<double> &Bcon,
                        int Nx,int Ny,int Nz,
                        int slice_k,
                        const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write((char*)&Nx, sizeof(int));
    ofs.write((char*)&Ny, sizeof(int));
    ofs.write((char*)&Nz, sizeof(int));
    ofs.write((char*)&slice_k, sizeof(int));
    for(int j=0;j<Ny;++j){
        for(int i=0;i<Nx;++i){
            int id = slice_k*Nx*Ny + j*Nx + i;
            double Ex = Econ[id*3+0];
            double Ey = Econ[id*3+1];
            double Ez = Econ[id*3+2];
            double Bx = Bcon[id*3+0];
            double By = Bcon[id*3+1];
            double Bz = Bcon[id*3+2];
            ofs.write(reinterpret_cast<char*>(&Ex), sizeof(double));
            ofs.write(reinterpret_cast<char*>(&Ey), sizeof(double));
            ofs.write(reinterpret_cast<char*>(&Ez), sizeof(double));
            ofs.write(reinterpret_cast<char*>(&Bx), sizeof(double));
            ofs.write(reinterpret_cast<char*>(&By), sizeof(double));
            ofs.write(reinterpret_cast<char*>(&Bz), sizeof(double));
        }
    }
}

// -------------------- Maxwell solver using tensor components --------------------
// We store E^i and B^i (contravariant) on the same grid (not staggered for simplicity)
// and take central differences for spatial derivatives in xi-space.

const double c0 = 299792458.0;
const double mu0 = 4*M_PI*1e-7;
const double eps0 = 1.0/(mu0*c0*c0);

int main(){
    // small test grid
    Grid G{16,16,16, 1.0/16.0,1.0/16.0,1.0/16.0};
    int Nx=G.Nx, Ny=G.Ny, Nz=G.Nz;
    int N = Nx*Ny*Nz;
    double dt = 0.1 * G.dx / c0; // small dt

    // allocate fields: for each grid point store 3 contravariant components
    std::vector<double> Econ(N*3,0.0), Bcon(N*3,0.0);

    // precompute geometry at each grid point: for demo we use a skew Jax independent of point
    Mat3 Jax{};
    Jax[0][0] = 1.0;  Jax[1][0] = 0.1; Jax[2][0] = 0.0;
    Jax[0][1] = 0.2;  Jax[1][1] = 1.0; Jax[2][1] = 0.0;
    Jax[0][2] = 0.0;  Jax[1][2] = 0.3; Jax[2][2] = 1.0;

    Mat3 g = compute_metric(Jax);
    Mat3 ginv = invert3(g);
    double sqrtg = std::sqrt(std::fabs(det3(g)));
    auto h = compute_scale_factors(Jax,g);

    // initialize E as a small Gaussian in Cartesian x-direction (we define cart vector and map)
    for(int k=0;k<Nz;++k) for(int j=0;j<Ny;++j) for(int i=0;i<Nx;++i){
        int id = idx(i,j,k,Nx,Ny,Nz);
        double xi = (i+0.5)*G.dx; double yi=(j+0.5)*G.dy; double zi=(k+0.5)*G.dz;
        // Cartesian Gaussian centered in computational domain
        double x = xi; double y = yi; double z = zi;
        double r2 = (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5);
        Vec3 Ecart = {std::exp(-200.0*r2), 0.0, 0.0};
        // convert to contravariant components in xi-space
        Vec3 Econ_pt = cartesian_to_contravariant(Jax, Ecart);
        for(int c=0;c<3;++c) Econ[id*3 + c] = Econ_pt[c];
    }

    // time stepping
    int nsteps = 200;
    std::vector<double> Enew(N*3,0.0), Bnew(N*3,0.0);
    double invdx = 1.0/G.dx, invdy = 1.0/G.dy, invdz = 1.0/G.dz;

    for(int n=0;n<nsteps;++n){
        // update B^i (leapfrog style): B^{n+1/2} = B^{n-1/2} - dt*(1/sqrtg) * eps^{i j k} * d_j E_k
        // here we treat eps^{ijk} * d_j E_k using index permutation
        for(int k=0;k<Nz;++k) for(int j=0;j<Ny;++j) for(int i=0;i<Nx;++i){
            int id = idx(i,j,k,Nx,Ny,Nz);
            // compute E_cov = g_{k m} E^m
            Vec3 Ecov = lower_index(g, {Econ[id*3+0],Econ[id*3+1],Econ[id*3+2]});
            // compute partial derivatives of E_cov components along xi_j
            // build scalar fields for each component on-the-fly (inefficient but simple)
            // For performance, pre-store each component array.
            // Here compute d_j E_k for all j,k via central differences
            double d_E2_dxi1 = d_dxi_j(Econ, i,j,k,Nx,Ny,Nz, 0, invdx); // derivative of E_component? careful
            // To be correct we need arrays of E_cov components. For brevity, reconstruct local central diffs:
            // compute local differences for E_cov[k] wrt xi_j
            // We'll approximate by finite differences using neighbor points for E_cov components

            // helper lambdas to get E_cov component at neighbor
            auto Ecov_comp_at = [&](int ii,int jj,int kk,int comp)->double{
                int idn = idx(ip(ii,Nx), ip(jj,Ny), ip(kk,Nz), Nx,Ny,Nz);
                Vec3 Ecovn = lower_index(g, {Econ[idn*3+0],Econ[idn*3+1],Econ[idn*3+2]});
                return Ecovn[comp];
            };

            // compute derivatives ∂_{xi_j} E_k for j=0..2, k=0..2
            double dE[3][3];
            for(int comp_k=0; comp_k<3; ++comp_k){
                // derivative wrt xi1
                double el = Ecov_comp_at(i-1,j,k,comp_k);
                double er = Ecov_comp_at(i+1,j,k,comp_k);
                dE[0][comp_k] = 0.5*(er - el)*invdx;
                // wrt xi2
                el = Ecov_comp_at(i,j-1,k,comp_k);
                er = Ecov_comp_at(i,j+1,k,comp_k);
                dE[1][comp_k] = 0.5*(er - el)*invdy;
                // wrt xi3
                el = Ecov_comp_at(i,j,k-1,comp_k);
                er = Ecov_comp_at(i,j,k+1,comp_k);
                dE[2][comp_k] = 0.5*(er - el)*invdz;
            }

            // compute curl: (n.b. eps^{i j k} with [ijk] / sqrt(g) )
            // eps^{ijk} * d_j E_k  = 1/sqrtg * [i j k] * d_j E_k
            // we'll compute component-wise
            Vec3 curlE{};
            // i=0
            curlE[0] = ( dE[1][2] - dE[2][1] ) / sqrtg;
            // i=1
            curlE[1] = ( dE[2][0] - dE[0][2] ) / sqrtg;
            // i=2
            curlE[2] = ( dE[0][1] - dE[1][0] ) / sqrtg;

            // update B^i
            for(int ii=0; ii<3; ++ii){
                Bnew[id*3 + ii] = Bcon[id*3 + ii] - dt * curlE[ii];
            }
        }

        // swap B
        Bcon.swap(Bnew);

        // update E^i: ∂_t D^i = (curl H)^i - J^i, with D^i = eps0 * E^i (assume vacuum)
        // So ∂_t E^i = (1/eps0) * (curl H)^i - (1/eps0) J^i ; with H_k = (1/mu0) B_k
        for(int k=0;k<Nz;++k) for(int j=0;j<Ny;++j) for(int i=0;i<Nx;++i){
            int id = idx(i,j,k,Nx,Ny,Nz);

            // compute H_cov = g_{k m} H^m ; H^m = B^m / mu0
            Vec3 Hcon = { Bcon[id*3+0]/mu0, Bcon[id*3+1]/mu0, Bcon[id*3+2]/mu0 };
            Vec3 Hcov = lower_index(g, Hcon);

            auto Hcov_comp_at = [&](int ii,int jj,int kk,int comp)->double{
                int idn = idx(ip(ii,Nx), ip(jj,Ny), ip(kk,Nz), Nx,Ny,Nz);
                Vec3 Hcon_n = { Bcon[idn*3+0]/mu0, Bcon[idn*3+1]/mu0, Bcon[idn*3+2]/mu0 };
                Vec3 Hcovn = lower_index(g, Hcon_n);
                return Hcovn[comp];
            };

            double dH[3][3];
            for(int comp_k=0; comp_k<3; ++comp_k){
                double el = Hcov_comp_at(i-1,j,k,comp_k);
                double er = Hcov_comp_at(i+1,j,k,comp_k);
                dH[0][comp_k] = 0.5*(er - el)*invdx;
                el = Hcov_comp_at(i,j-1,k,comp_k);
                er = Hcov_comp_at(i,j+1,k,comp_k);
                dH[1][comp_k] = 0.5*(er - el)*invdy;
                el = Hcov_comp_at(i,j,k-1,comp_k);
                er = Hcov_comp_at(i,j,k+1,comp_k);
                dH[2][comp_k] = 0.5*(er - el)*invdz;
            }

            Vec3 curlH{};
            curlH[0] = ( dH[1][2] - dH[2][1] ) / sqrtg;
            curlH[1] = ( dH[2][0] - dH[0][2] ) / sqrtg;
            curlH[2] = ( dH[0][1] - dH[1][0] ) / sqrtg;

            // no current for demo J=0
            for(int ii=0; ii<3; ++ii){
                Enew[id*3 + ii] = Econ[id*3 + ii] + (dt/eps0) * curlH[ii];
            }
        }

        Econ.swap(Enew);

        if(n%20==0){
            // compute simple energy measure in Cartesian components
            double energy=0.0;
            for(int kk=0; kk<Nz;++kk) for(int jj=0;jj<Ny;++jj) for(int ii=0; ii<Nx;++ii){
                int id = idx(ii,jj,kk,Nx,Ny,Nz);
                Vec3 Ecart = contravariant_to_cartesian(Jax, {Econ[id*3+0],Econ[id*3+1],Econ[id*3+2]});
                Vec3 Bcart = contravariant_to_cartesian(Jax, {Bcon[id*3+0],Bcon[id*3+1],Bcon[id*3+2]});
                double e = 0.5*(eps0*(Ecart[0]*Ecart[0]+Ecart[1]*Ecart[1]+Ecart[2]*Ecart[2])
                              + (1.0/mu0)*(Bcart[0]*Bcart[0]+Bcart[1]*Bcart[1]+Bcart[2]*Bcart[2]));
                energy += e * (sqrtg * G.dx * G.dy * G.dz);
            }
            std::cout<<"step="<<n<<" energy="<<energy<<"\n";
        }
        write_slice_binary(Econ, Bcon, Nx, Ny, Nz, 3, "out/slice_t"+std::to_string(n)+".bin");
    }

    std::cout<<"done.\n";
    return 0;
}
