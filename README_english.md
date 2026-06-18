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

---

# Usage Guide

The remainder of this document explains how to configure and run simulations using the framework.

As a concrete example, we use:

```text
main/whistler_kappa_super.cpp
```

which solves a Whistler-wave problem using cylindrical velocity coordinates and MPI parallelization.

The following sections describe:

1. Axis definitions
2. Tensor definitions
3. Coordinate mappings
4. Jacobian implementation
5. Boundary conditions
6. Advection terms
7. Solver construction
8. Time integration
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
class Physic_vx
{
    ...
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
