# Generic-Vlasov-Solver

A generic, high-performance Vlasov solver for kinetic plasma simulations in arbitrary dimensions and coordinate systems.

> **Notice**
>
> This repository is a public extract of a larger private development repository used for active development. Commit history and certain internal resources have been intentionally omitted from this public release.

---

# Overview

This project is a highly flexible Vlasov simulation framework designed to perform kinetic plasma simulations in arbitrary dimensions and coordinate systems.

Using C++ template metaprogramming, the computational domain—including coordinate axes, grid sizes, boundary conditions, and MPI domain decomposition—is fully abstracted at compile time. As a result, a wide variety of plasma-wave simulations can be implemented with minimal modifications to the core codebase.

The framework solves the following equations:

* The **Vlasov equation**
* **Maxwell's equations**

Maxwell's equations are implemented using a conventional Finite-Difference Time-Domain (FDTD) method.

The solver supports a broad range of applications, from simple 1D1V electrostatic simulations to large-scale MPI-parallel simulations on supercomputers involving high-dimensional phase spaces and complex coordinate systems such as cylindrical or dipole coordinates.

---

# Motivation and Background

This project was originally developed to investigate kinetic plasma processes in near-Earth space environments, including:

* Aurora generation mechanisms
* Wave-particle interactions in the magnetosphere
* Space-weather forecasting applications
* Radiation exposure prediction for spacecraft and astronauts

In plasma simulations, the **Particle-In-Cell (PIC)** method is the most widely used approach. PIC simulations track a large number of individual particles and reconstruct macroscopic quantities from their collective behavior.

While powerful, PIC methods suffer from an inherent limitation: because only a finite number of simulation particles can be used, the results inevitably contain **statistical noise**.

To overcome this issue, this project adopts the **Vlasov method**, which evolves the particle velocity distribution function directly rather than tracking individual particles.

As a consequence:

* Statistical noise is completely eliminated.
* Fine structures in phase space can be resolved accurately.
* Weak wave-particle interactions become easier to analyze.
* Long-time evolution can be studied without particle noise contamination.

---

# Author's Role

The entire framework was developed independently, including:

* Core architecture design
* Template-metaprogramming abstractions
* Automatic MPI communication topology generation
* Numerical solver implementation
* Verification simulations
* Physics applications

Example validation cases include:

* Langmuir waves
* Two-stream instability
* Bernstein modes
* Whistler waves

---

# Technology Stack and Architecture

## Language

* C++17

## Parallel Computing

* MPI (Message Passing Interface)

## Design Philosophy

### Compile-Time Abstraction

Dimensions, grid sizes, boundary conditions, and parallel decomposition parameters are specified at compile time using C++ templates.

This design eliminates runtime overhead while maintaining flexibility.

### Separation of Physics and Numerical Infrastructure

Physical operators and advection terms are encapsulated independently:

* `Operators`
* `Advections`

Both are managed through generic `Pack` classes, allowing numerical schemes and physical models to be combined flexibly.

### Automatic MPI Topology Generation

MPI communication patterns are generated automatically from the information provided to the `Axis` template:

```cpp
Axis<ID, LocalGrids, MPI_Procs, GhostCells>
```

Using the specified:

* domain decomposition,
* ghost-cell width,
* boundary conditions,

the framework automatically constructs the communication graph required for ghost-cell exchange.

As a result, users can implement simulations without manually writing MPI communication code.

---

# Numerical Challenge and Solution

## Problem: Numerical Diffusion During Cyclotron Motion

A major challenge in conventional Cartesian-coordinate Vlasov simulations arises when modeling charged particles in magnetic fields.

Electrons undergo rapid cyclotron motion, continuously rotating in velocity space.

In Cartesian velocity coordinates, repeated interpolation during this rotation introduces numerical diffusion. Over time, the distribution function becomes increasingly blurred and distorted, eventually destroying important kinetic structures.

This issue is illustrated below:

```text
[Original README Figure:
Distribution function degradation caused by numerical diffusion]
```

---

## Proposed Solution: Polar Velocity Coordinates

The key observation is that cyclotron motion is fundamentally a rotational motion.

Instead of representing velocity space using Cartesian coordinates:

```math
(v_x, v_y)
```

the solver introduces polar coordinates:

```math
(v_r, v_\theta)
```

In this representation, circular particle motion becomes a simple translation in the angular direction.

Consequently:

* Rotational dynamics are represented naturally.
* Radial numerical diffusion is eliminated.
* Long-term preservation of the distribution function is achieved.
* Cyclotron motion can be simulated with significantly higher accuracy.

The improvement is shown in the original README:

```text
[Original README Figure:
Distribution function preserved using polar velocity coordinates]
```

Even after long simulation times, the shape of the distribution function remains nearly unchanged.

---

# What is the Vlasov Equation?

The Vlasov equation describes the evolution of a collisionless plasma using a distribution function rather than individual particles.

---

## 1. Distribution Function

In phase space, we define the distribution function

```math
f(\mathbf{x}, \mathbf{v}, t)
```

such that

```math
f(\mathbf{x}, \mathbf{v}, t)
\, d\mathbf{x}
\, d\mathbf{v}
```

represents the number of particles contained within an infinitesimal volume around position

```math
\mathbf{x}
```

and velocity

```math
\mathbf{v}
```

at time

```math
t.
```

---

## 2. Governing Equation

For a collisionless plasma under electromagnetic forces, the Vlasov equation is

```math
\frac{\partial f}{\partial t}
+
\mathbf{v}\cdot\nabla_x f
+
\frac{q}{m}
(\mathbf{E}+\mathbf{v}\times\mathbf{B})
\cdot\nabla_v f
=
0
```

where:

| Symbol       | Meaning                 |
| ------------ | ----------------------- |
| (f)          | Distribution function   |
| (q)          | Particle charge         |
| (m)          | Particle mass           |
| (\mathbf{E}) | Electric field          |
| (\mathbf{B}) | Magnetic field          |
| (\mathbf{v}) | Velocity                |
| (\nabla_x)   | Spatial gradient        |
| (\nabla_v)   | Velocity-space gradient |

---

## 3. Physical Interpretation

The equation consists of three terms.

### Time Evolution

```math
\frac{\partial f}{\partial t}
```

Represents the temporal change of the distribution function at a fixed observation point.

### Spatial Advection

```math
\mathbf{v}\cdot\nabla_x f
```

Represents changes caused by particle motion through physical space.

### Velocity-Space Advection

```math
\frac{q}{m}
(\mathbf{E}+\mathbf{v}\times\mathbf{B})
\cdot\nabla_v f
```

Represents changes in velocity caused by the Lorentz force.

---

# Key Features

## Flexible Dimension Extension

Any number of dimensions, grid sizes, ghost-cell widths, and boundary conditions can be specified simply by defining `Axis` template classes.

No modifications to the core solver are required.

---

## Type-Safe High-Dimensional Tensors

The framework provides:

```cpp
NdTensorWithGhostCell
```

for efficient and type-safe storage of:

* distribution functions
* electromagnetic fields
* Jacobian components
* auxiliary quantities

in arbitrary dimensions.

---

## Separation of Operators and Advections

Physical operators (`Operators`) and advection terms (`Advections`) are encapsulated independently through generic `Pack` classes.

This architecture allows new numerical schemes and physical models to be integrated with minimal effort.

---

## MPI Parallelization Support

Parallel decomposition can be specified independently for each axis:

```cpp
Axis<ID, LocalGrids, MPI_Procs, GhostCells>
```

The framework automatically generates the communication graph and ghost-cell exchange logic according to the chosen boundary conditions.

Users can therefore focus on physics implementation rather than MPI programming.

---

## Support for General Coordinate Systems

The framework separates:

* Computational coordinates
* Physical coordinates

Any coordinate system can be used as long as a one-to-one mapping exists between the two spaces.

The user only needs to provide:

* Coordinate transformation classes
* Jacobian components
* Metric tensor components

Examples include:

* Cartesian coordinates
* Cylindrical coordinates
* Spherical coordinates
* Dipole coordinates
* User-defined coordinate systems
# Included Simulation Examples

The repository contains several validation and application examples demonstrating the flexibility of the framework across different dimensions, coordinate systems, and physical phenomena.

---

## 1. `1D0V.cpp`

### Purpose

The simplest example included in the repository.

This program solves a one-dimensional advection problem and is primarily intended for:

* Testing boundary conditions
* Verifying advection schemes
* Debugging framework modifications
* Benchmarking new Vlasov schemes

Because of its simplicity, it is often the first example used when implementing a new numerical method.

### Example Result

```text
[Original README Figure:
1D advection simulation]
```

---

## 2. `langmuir_wave.cpp`

### Configuration

* 1D in physical space
* 1D in velocity space (1D1V)
* Cartesian coordinates

### Physical Problem

This example simulates the linear evolution of a Langmuir wave.

Langmuir waves are electrostatic oscillations of electrons in a plasma and serve as one of the most fundamental validation problems for kinetic plasma solvers.

### Verification

The electric field

```math
E(x,t)
```

is Fourier transformed into frequency–wavenumber space:

```math
(\omega, k)
```

The resulting spectrum reproduces the theoretical Langmuir-wave dispersion relation.

### Example Result

```text
[Original README Figure:
Langmuir-wave dispersion relation]
```

---

## 3. `two_stream_instability.cpp`

### Configuration

* 1D in physical space
* 1D in velocity space (1D1V)
* Cartesian coordinates

### Physical Problem

This example simulates the classic Two-Stream Instability.

Two electron populations are initialized with different drift velocities. Small perturbations grow exponentially due to kinetic instability.

### Phenomena Observed

The simulation reproduces:

* Linear instability growth
* Wave amplification
* Nonlinear saturation
* Formation of phase-space vortices
* Formation of phase-space holes

These phase-space structures are important signatures of nonlinear kinetic plasma dynamics.

### Example Result

```text
[Original README Video:
Phase-space hole formation]
```

---

## 4. `bernstein_mode_wave_super.cpp`

### Configuration

* 1D in physical space
* 2D in velocity space (1D2V)
* Polar velocity coordinates
* MPI-parallel implementation

### Physical Problem

This simulation models electrostatic waves propagating perpendicular to a background magnetic field.

In magnetized plasmas, such waves excite Bernstein modes through cyclotron harmonics.

### Why Polar Coordinates?

Cyclotron motion is naturally represented in polar velocity coordinates.

This allows the solver to:

* Eliminate radial numerical diffusion
* Preserve the velocity-space distribution
* Resolve high-order cyclotron harmonics accurately

### Verification

The electric field spectrum is transformed into

```math
(\omega,k)
```

space.

The resulting spectrum reproduces the theoretical Bernstein-mode dispersion relation.

### Example Result

```text
[Original README Figure:
Bernstein-mode dispersion relation]
```

### Parallel Computing

A fully MPI-parallel version is included for large-scale simulations on supercomputers.

---

## 5. `whistler_cylinder_super.cpp`

### Configuration

* 1D in physical space
* 3D in velocity space (1D3V)
* Cylindrical velocity coordinates
* MPI-parallel implementation

### Physical Problem

This example simulates Whistler-mode waves in inhomogeneous and anisotropic plasmas.

The simulation includes:

* Wave propagation
* Wave growth
* Wave-particle interactions
* Resonant electron dynamics

### Scientific Importance

Whistler waves play a major role in:

* Radiation-belt dynamics
* Electron acceleration
* Pitch-angle scattering
* Auroral processes
* Magnetospheric plasma transport

Because of this, Whistler-wave simulations are an important benchmark for kinetic plasma solvers.

### Verification

The simulation reproduces:

#### Dispersion Relation

The computed spectrum agrees with the theoretical Whistler-wave dispersion relation.

#### Linear Growth Rate

The measured wave growth rate agrees with linear kinetic theory.

#### Growth Rate as a Function of Wavenumber

The dependence

```math
\gamma(k)
```

matches theoretical predictions.

### Example Results

```text
[Original README Figures:
Whistler-wave dispersion relation
Growth-rate comparison
γ(k) comparison]
```

### Parallel Computing

This example is specifically designed for large-scale MPI execution on supercomputers.

The phase space is decomposed across multiple dimensions, allowing simulations with thousands of MPI processes.

---

# Software Requirements

The framework has very few external dependencies.

## Compiler

Any compiler supporting C++17 or newer:

* GCC
* Clang
* Intel Compiler
* Other C++17-compliant compilers

## MPI Library

An MPI implementation is required for parallel execution:

* OpenMPI
* MPICH
* Intel MPI
* Other MPI-compatible implementations

---

# Mathematical Framework

The framework is designed around a complete separation between:

* Computational space
* Physical space

The numerical solver operates exclusively on a uniform computational grid.

Users define a mapping

```math
(i,j,k,\ldots)
\rightarrow
(x,y,z,v_x,v_y,v_z,\ldots)
```

that connects computational coordinates to physical coordinates.

By providing:

* Coordinate transformation classes
* Jacobian matrices
* Metric tensor components

the same solver can be applied to virtually any coordinate system.

This design enables simulations in:

* Cartesian coordinates
* Cylindrical coordinates
* Spherical coordinates
* Dipole coordinates
* Custom user-defined coordinates

without modifying the underlying numerical infrastructure.

# Usage Guide

This framework is designed to allow users to define arbitrary dimensions and coordinate systems at compile time. By completely separating the computational space (uniform grids used by the numerical solver) from the physical space (actual coordinates used in the physical model), complex coordinate systems can be implemented simply by providing the appropriate coordinate transformations and Jacobian components.

The following sections explain the basic workflow using:

```text
main/whistler_kappa_super.cpp
```

as an example.

---

# 1. Defining Dimensions, Grid Sizes, and MPI Decomposition

The dimensionality of a simulation is determined by the collection of `Axis` definitions.

One of the key design goals of this framework is that dimensions can be added or removed without modifying the core solver. Extending a simulation from 1D1V to 1D3V, for example, simply requires defining additional axes.

Each computational axis is defined using the `Axis` template:

```cpp
// Axis<ID, LocalGridCount, MPIProcessCount, GhostCellCount>

using Axis_z_ = Axis<0, 512 / 32, 32, 3>; // Spatial z-axis
using Axis_vr = Axis<1, 256 / 16, 16, 3>; // Velocity r-axis
using Axis_vt = Axis<2,  64 /  4,  4, 3>; // Velocity theta-axis
using Axis_vp = Axis<3,  64 /  1,  1, 3>; // Velocity phi-axis
```

## Parameters

### Axis ID

The first parameter is a unique integer identifier beginning from zero.

```cpp
Axis<0, ...>
Axis<1, ...>
Axis<2, ...>
```

Lower IDs correspond to outer dimensions in memory layout.

When adding a new dimension, simply assign a new unused ID.

---

### Local Grid Count

The second parameter specifies the number of grid points handled by a single MPI process.

For example:

```cpp
Axis<0, 512 / 32, 32, 3>
```

means:

* Total global grid count = 512
* Number of MPI processes = 32
* Local grid count = 16

---

### MPI Process Count

The third parameter specifies how many MPI processes are assigned to the axis.

In the above example:

```cpp
32 × 16 × 4 = 2048
```

MPI processes are used in total.

---

### Ghost Cell Count

The fourth parameter specifies the number of ghost cells used on both sides of the domain.

Ghost cells are required by high-order advection schemes and MPI communication.

---

## Resulting Phase-Space Layout

For the example above:

```cpp
Axis_z_
Axis_vr
Axis_vt
Axis_vp
```

each MPI process owns a local block of size:

```text
[16, 16, 16, 64]
```

while the total number of MPI ranks is:

```text
2048
```

---

# 2. Defining Containers for Physical Quantities

Once the computational axes are defined, tensor types can be created using:

```cpp
NdTensorWithGhostCell
```

This class automatically generates multidimensional containers with:

* arbitrary dimensions
* arbitrary ghost-cell widths
* MPI-aware indexing

---

## 2.1 Distribution Function

The distribution function

```math
f(z,v_r,v_\theta,v_\phi)
```

is a scalar quantity.

Therefore its element type is simply:

```cpp
double
```

The corresponding tensor definition is:

```cpp
using Value = double;

using DistributionFunction =
    NdTensorWithGhostCell<
        Value,
        Axis_z_,
        Axis_vr,
        Axis_vt,
        Axis_vp
    >;
```

Since the distribution function depends on both physical space and velocity space, all four axes are included.

This produces a four-dimensional tensor of scalar values.

---

## 2.2 Electromagnetic Fields

Unlike the distribution function, electromagnetic fields are vector quantities.

The framework provides:

```cpp
Vec3<Value>
```

for storing vector data.

### Magnetic Field

```cpp
#include "../vec3.h"

using MagneticField =
    NdTensorWithGhostCell<
        Vec3<Value>,
        Axis_z_
    >;
```

---

### Electric Field

```cpp
using ElectricField =
    NdTensorWithGhostCell<
        Vec3<Value>,
        Axis_z_
    >;
```

Because the fields depend only on physical space, only the spatial axis is required.

This produces a one-dimensional tensor whose elements are three-dimensional vectors.

---

## Yee Grid Arrangement

Maxwell's equations are solved using the Yee FDTD scheme.

Consequently, electric and magnetic fields are staggered in both space and time.

For example:

```cpp
B(i,j).z = Bz(x = Δx i      , t = Δt(j+1/2))
B(i,j).x = Bx(x = Δx(i+1/2), t = Δt(j+1/2))
B(i,j).y = By(x = Δx(i+1/2), t = Δt(j+1/2))
```

while

```cpp
E(i,j).z = Ez(x = Δx(i+1/2), t = Δt j)
E(i,j).x = Ex(x = Δx i      , t = Δt j)
E(i,j).y = Ey(x = Δx i      , t = Δt j)
```

This staggering preserves second-order accuracy and naturally satisfies discrete Maxwell equations.

---

# 3. Defining Coordinate Systems

The numerical solver operates entirely in computational coordinates.

Therefore, users must define a mapping between:

```math
(i,j,k,\ldots)
```

and

```math
(z,v_x,v_y,v_z,\ldots)
```

The mapping must be one-to-one.

In addition, users provide:

* Jacobian components
* Metric tensor components

required for coordinate transformations.

In the Whistler-wave example:

* Physical space uses a Cartesian coordinate.
* Velocity space uses cylindrical (polar) coordinates.

---

# 3.1 Mapping Computational Coordinates to Physical Coordinates

## 3.1.1 Mapping to Physical Coordinate z

A simple example is the spatial coordinate:

```cpp
class CalcZ__2_Z_
{
private:
    const int z__start_id;

    static int calc_start_id(const int my_world_rank)
    {
        auto [axis_z_,
              axis_vr,
              axis_vt,
              axis_vp]
        =
        axis_instantiator<
            Axis_z_,
            Axis_vr,
            Axis_vt,
            Axis_vp
        >(my_world_rank);

        return axis_z_.L_id;
    }

public:

    CalcZ__2_Z_(const int my_world_rank)
        :
        z__start_id(calc_start_id(my_world_rank))
    {}

    Value at(const int calc_z_) const
    {
        return Global::grid_size_z_
            * (0.5 + (double)(z__start_id + calc_z_));
    }
};
```

Because the physical grid is uniform, the coordinate is obtained by multiplying the global index by the grid spacing.

Notice that:

```cpp
calc_z_
```

is a local index.

Therefore the starting global index of the current MPI process must be added.

---

### Physical Coordinate Wrapper

```cpp
class Physic_z_
{
    const CalcZ__2_Z_ calc_z__2_z;

public:

    Physic_z_(const int my_world_rank)
        :
        calc_z__2_z(my_world_rank)
    {}

    Value honestly_translate(
        const int calc_z,
        const int calc_vr,
        const int calc_vt
    ) const
    {
        return calc_z__2_z.at(calc_z);
    }

    Value at(
        const int calc_z,
        const int calc_vr,
        const int calc_vt,
        const int calc_vp
    ) const
    {
        return calc_z__2_z.at(calc_z);
    }

    static constexpr int label = 0;
};
```

The first physical coordinate is assigned:

```cpp
label = 0
```

The helper class `CalcZ__2_Z_` is optional but often improves readability.

---

## 3.1.2 Mapping to Physical Coordinate vx

Velocity coordinates are more complicated because the computational coordinates are:

```math
(v_r,v_\theta,v_\phi)
```

while the physical coordinates are:

```math
(v_x,v_y,v_z)
```

The transformation is:

```math
v_x
=
v_r \sin(v_\theta)\cos(v_\phi)
```

A corresponding implementation is:

```cpp
// 例：計算空間 (vr, vt, vp) から 物理空間 vx への写像 (vx = vr * sin(vt) * cos(vp))

class Physic_vx
{
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt,Axis_vp> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt,const int calc_vp)const{
        // v_x = vr * cos(vt)
        const Value vr = calc_vr_2_vr.at(calc_vr);
        const Value vt = calc_vt_2_vt.at(calc_vt);
        const Value vp = calc_vp_2_vp.at(calc_vp);
        return vr * sin(vt)*cos(vp);
    }

    Physic_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank),
        calc_vp_2_vp(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t,FullSliceGhost_p>(
            [this](const int calc_vr,const int calc_vt,const int calc_vp){
                return honestly_translate(calc_vr, calc_vt,calc_vp);
            }
        );
    }
    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);    
    }
    static const int label = 1;
};

```

(identical to the source code example)

---

### Why Use a Lookup Table?

Direct evaluation requires expensive operations such as:

```cpp
sin()
cos()
```

for every grid point.

To reduce computational cost, the values are precomputed once and stored in:

```cpp
NdTensorWithGhostCell
```

which acts as a lookup table.

This significantly improves performance during time integration.

---

### Labels

The physical coordinates are assigned labels:

```cpp
Physic_z_  -> label = 0
Physic_vx  -> label = 1
Physic_vy  -> label = 2
Physic_vz  -> label = 3
```

The labels determine the ordering of coordinates within the Jacobian and metric tensor framework.

---

The same procedure is used to implement:

```cpp
Physic_vy
Physic_vz
```

using the appropriate coordinate transformations.
# 3.2 Implementation of Metric Tensors Between Computational and Physical Spaces

A lightweight tensor can be written as:

```math
\frac{\partial(z_c,v_r,v_\theta,v_\phi)}
     {\partial(z,v_x,v_y,v_z)}
=
\begin{pmatrix}
\frac{\partial z_c}{\partial z} & 0 & 0 & 0 \\
0 & \frac{\partial v_r}{\partial v_x} & \frac{\partial v_r}{\partial v_y} & \frac{\partial v_r}{\partial v_z} \\
0 & \frac{\partial v_\theta}{\partial v_x} & \frac{\partial v_\theta}{\partial v_y} & \frac{\partial v_\theta}{\partial v_z} \\
0 & \frac{\partial v_\phi}{\partial v_x} & \frac{\partial v_\phi}{\partial v_y} & 0
\end{pmatrix}
```

We now implement classes that represent each element of this matrix.

## 3.2.1 Implementation of Metric Tensor Elements

As an example, the following class represents the tensor element

```math
\frac{\partial v_r}{\partial v_x}.
```

The member function `.at()` must return the corresponding value. Its arguments are the four local grid indices in computational space. Adjust the number of arguments according to the dimensionality of your problem.

```cpp


class Vr_diff_vx
{
private:
    NdTensorWithGhostCell<Value,Axis_vt,Axis_vp> table;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;
public:
    Value honestly_translate(const int calc_vt,const int calc_vp){
        // sinθ cosφ
        const Value vt = calc_vt_2_vt.at(calc_vt);
        const Value vp = calc_vp_2_vp.at(calc_vp);
        return sin(vt) * cos(vp)/(double)Global::grid_size_vr;
    }
    Vr_diff_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vt_2_vt(my_world_rank),
        calc_vp_2_vp(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_t,FullSliceGhost_p>(
            [this](const int calc_vt,const int calc_vp){
                return honestly_translate(calc_vt, calc_vp);
            }
        );
    }
    
    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vt,calc_vp);
    }
};
```

Since the values are precomputed and accessed through a lookup table (LUT), improved performance can be expected. Implement the remaining tensor elements in the same manner.

## 3.2.2 Instantiation of the Metric Tensor

Instantiate the Metric tensor inside `main()` as follows:

```cpp

const Independent independent;
const Z__diff_z_ z__diff_z_;
const Vr_diff_vx vr_diff_vx(world_rank);
const Vr_diff_vy vr_diff_vy(world_rank);
const Vr_diff_vz vr_diff_vz(world_rank);
const Vt_diff_vx vt_diff_vx(world_rank);
const Vt_diff_vy vt_diff_vy(world_rank);
const Vt_diff_vz vt_diff_vz(world_rank);
const Vp_diff_vx vp_diff_vx(world_rank);
const Vp_diff_vy vp_diff_vy(world_rank);

const Jacobian jacobian(
   z__diff_z_ , independent, independent, independent, 
   independent, vr_diff_vx , vr_diff_vy , vr_diff_vz ,
   independent, vt_diff_vx , vt_diff_vy , vt_diff_vz ,
   independent, vp_diff_vx , vp_diff_vy , independent
);
```

For entries whose derivative is zero, use `independent`.

---

# 4. Setting Boundary Conditions

Boundary conditions define how ghost cells are updated at the boundaries of the computational domain.

In this framework, MPI communication and ghost-cell updates are automatically handled once the index mapping (i.e., which cell is copied into each ghost cell) is specified.

Define `left` and `right` template functions inside the boundary-condition class, and specify the source coordinates using global indices.

## 4.1 Example: Periodic Boundary Condition (z-axis)

This is the simplest example of a periodic boundary condition. Values are copied from the opposite side of the domain.

```cpp

class BoundaryCondition_z_
{
public:
    static const int label = 0;

    // 吸収境界など、単なる値のコピー以外の特殊な処理を行う場合はtrueにしますが、通常はfalseです。
    static constexpr bool not_only_comm = false;

    // 左側のゴーストセルを更新するためのコピー元インデックスを返す
    template<int Index>
    static int left(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            // z軸方向は右端（+ num_global_grid）のインデックスを参照
            return calc_z + Axis_z_::num_global_grid;
        }
        else if constexpr(Index == 1){ return calc_vr; }
        else if constexpr(Index == 2){ return calc_vt; }
        else if constexpr(Index == 3){ return calc_vp; }
        else return 0;
    }

    // 右側のゴーストセルを更新するためのコピー元インデックスを返す
    template<int Index>
    static int right(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            // z軸方向は左端（- num_global_grid）のインデックスを参照
            return calc_z - Axis_z_::num_global_grid;
        }
        else if constexpr(Index == 1){ return calc_vr; }
        else if constexpr(Index == 2){ return calc_vt; }
        else if constexpr(Index == 3){ return calc_vp; }
        else return 0;
    }
};
```

### Explanation

* `left()` returns the source index used to update the left ghost cells.
* `right()` returns the source index used to update the right ghost cells.
* Along the z-direction, data are copied from the opposite end of the global domain.

## 4.2 Complex Boundary Conditions Such as Polar Coordinates (Reference)

The velocity space in `whistler_kappa_super.cpp` is defined in polar coordinates

```math
(v_r, v_\theta, v_\phi).
```

For example, when attempting to access a region with

```math
v_r < 0,
```

the coordinate must pass through the origin and refer to data at the opposite angular position:

```math
v_\theta \rightarrow \pi - v_\theta,
\qquad
v_\phi \rightarrow v_\phi + \pi.
```

Within this framework, such complex spatial topologies can be described concisely using conditional index mappings. See `BoundaryCondition_vr` and related source files for details.

## 4.3 Packing Boundary Conditions

Combine the boundary-condition classes for all axes into a single pack:

```cpp
using BoundaryCondition = Pack<
    BoundaryCondition_z_,
    BoundaryCondition_vr,
    BoundaryCondition_vt,
    BoundaryCondition_vp
>;
```

---

# 5. Definition of Advection Terms (Fluxes)

Define the advection terms appearing in each part of the Vlasov equation.

For

```math
\frac{\partial f}{\partial t}
+
\nabla \cdot (\mathbf{F}f)
=
0,
```

the quantity corresponding to

```math
\mathbf{F}
```

must be specified.

## 5.1 Example: Velocity-Space Advection Due to the Lorentz Force

The following example implements advection in the

```math
v_x
```

direction,

```math
-\frac{e}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})_x.
```

Because a Yee grid is used, the electric and magnetic fields are staggered by half a grid spacing and therefore require interpolation.

```cpp

class Fvx {
private:
    const bool _is_velo_right_edge;
    const ElectricField& e_field;
    const MagneticField& m_field;
    const Physic_vz& physic_vz;
    const Physic_vy& physic_vy;

public:
    Fvx(const int my_world_rank, const ElectricField& e_field, const MagneticField& m_field,
        const Physic_vz& physic_vz, const Physic_vy& physic_vy) : /* ...初期化... */ {}

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp) const {
        // 速度空間の境界で粒子が計算領域外に逃げないよう、フラックスを0にする処理
        if(_is_velo_right_edge && calc_vr == Axis_vr::num_grid){
            return - at(calc_z, Axis_vr::num_grid-1, calc_vt, calc_vp);
        }
        else{
            // Yee格子に基づき、必要な位置の電磁場を取得（補間）
            const Value Ex = e_field.at(calc_z).x;
            const Value By = (m_field.at(calc_z-1).y + m_field.at(calc_z).y)/2.;
            const Value Bz = m_field.at(calc_z).z;

            // 物理速度を取得
            const Value vz = physic_vz.at(calc_z, calc_vr, calc_vt, calc_vp);
            const Value vy = physic_vy.at(calc_z, calc_vr, calc_vt, calc_vp);

            // ローレンツ力 -(E + v×B)_x を計算（電子電荷が負のため - を付与）
            return - (Ex + vy*Bz - vz*By); 
        }
    }
};
```

### Explanation

* At the velocity-space boundary, the flux is forced to zero so that particles do not leave the computational domain.
* Electromagnetic fields are interpolated according to the Yee-grid arrangement.
* Physical velocities are obtained from the corresponding coordinate-conversion classes.
* The Lorentz force term is evaluated as

```math
-(E_x + v_yB_z - v_zB_y),
```

where the minus sign accounts for the negative charge of electrons.

After defining all advection terms, combine them into a pack:

```cpp
const Pack advections(
    flux_z_,
    flux_vx,
    flux_vy,
    flux_vz
);
```

---

# 6. Numerical Scheme and Overall Construction

Select a high-order interpolation scheme for flux calculations and construct the Vlasov solver by passing all previously defined components (distribution function, advection terms, Jacobian, numerical scheme, etc.) to `AdvectionEquation`.

```cpp

// 補間スキームの選択（例: Umeda 2008の3次精度スキーム）
#include "../schemes/umeda_2008.h"
using Scheme = Umeda2008;
namespace Global{ Scheme scheme; }

// AdvectionEquation (Vlasovソルバー) のインスタンス化
AdvectionEquation equation(world_rank, dist_function, advections, jacobian, Global::scheme, current);

// 境界条件管理マネージャーのインスタンス化
BoundaryManager boundary_manager(world_rank, world_size, dist_function, boundary_condition, axis_z_, axis_vr, axis_vt, axis_vp);
```

The example above uses the third-order scheme proposed by Umeda (2008).

A `BoundaryManager` instance is also created to manage boundary-condition updates and MPI synchronization.

---

# 7. Main Loop and Time Integration

Time advancement is performed using **Strang splitting**.

By solving the advection equation independently along each axis, the multidimensional problem is reduced to a sequence of one-dimensional problems.

After each call to

```cpp
equation.solve<Axis_**>(dt);
```

you must immediately call

```cpp
boundary_manager.apply<Axis_**>();
```

to synchronize ghost cells through MPI communication.

```cpp

for(int i=0; i<num_steps; i++){
    // 1. 速度空間の半ステップ進行 (dt/2)
    equation.solve<Axis_vr>(dt/2.); boundary_manager.apply<Axis_vr>();
    equation.solve<Axis_vt>(dt/2.); boundary_manager.apply<Axis_vt>();
    equation.solve<Axis_vp>(dt/2.); boundary_manager.apply<Axis_vp>();

    // 2. 電流のクリアと、実空間の1ステップ進行 (dt)
    current.clear();
    equation.solve<Axis_z_>(dt);    
    boundary_manager.apply<Axis_z_>();
    
    // 3. 電流の計算とMaxwell方程式 (FDTD) の進行
    current_calculator.calc();
    current.compute_global_current();
    // (FDTDソルバーによる電磁場の時間発展... 詳細はソース参照)

    // 4. 速度空間の残り半ステップ進行 (dt/2)
    equation.solve<Axis_vp>(dt/2.); boundary_manager.apply<Axis_vp>();
    equation.solve<Axis_vt>(dt/2.); boundary_manager.apply<Axis_vt>();
    equation.solve<Axis_vr>(dt/2.); boundary_manager.apply<Axis_vr>();

    // ※ 適宜、ログの書き出しやデータ保存を行う
```

### Time-Stepping Procedure

1. Advance the velocity-space directions by a half step (`dt/2`).
2. Clear the current and advance the physical-space direction by one full step (`dt`).
3. Compute the current and advance Maxwell's equations using the FDTD solver.
4. Advance the remaining half step in velocity space.
5. Output logs and save data as needed.

With these steps completed, the framework is ready to perform a general Vlasov simulation in an arbitrary coordinate system. Adjust the grid resolution, initial conditions (e.g., `init_dist_and_poisson`), and other physical parameters according to the problem of interest.
