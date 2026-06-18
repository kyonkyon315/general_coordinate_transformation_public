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
