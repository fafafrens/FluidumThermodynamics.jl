# Hadron resonance gas

`HadronResonanceGas` evaluates an ideal gas of user-supplied, positive-mass,
zero-width resonances. The particle list defines the model: degeneracy includes
the states represented by that entry, and antiparticles must be listed explicitly.
No spectrum, excluded-volume interaction, resonance-width integration, or decay
feed-down is implicit.

## Pressure and derivatives

For each species with mass `m`, degeneracy `g`, charges `q`, and
`μs = dot(q, μ)`, the pressure is

```math
p_s = \mathrm{unit\_scale}\,\frac{g m^2 T^2}{2\pi^2}
\sum_{k=1}^{K}\frac{c_k}{k^2}e^{k\mu_s/T}K_2(km/T).
```

Boltzmann statistics use `K=1, c₁=1`; bosons use `cₖ=1`; fermions use
`cₖ=(-1)^(k+1)`. All three use relativistic single-particle energies.
Quantum sums are truncated at the EOS `terms` (default 16), not automatically
converged. Both Bose and Fermi expansions here require `μs < m`. Increase
`terms` until pressure and the derivatives you need converge, especially near
this boundary; the implementation does not cover degenerate fermions above it
or Bose condensation.

The analytic gradient and Hessian use [modified-Bessel recurrences](https://dlmf.nist.gov/10.29).
Scaled Bessel functions from [SpecialFunctions.jl](https://specialfunctions.juliamath.org/stable/functions_list/#Bessel-Functions)
combine the fugacity and mass exponential before evaluation. This avoids
multiplying a vanishing Bessel value by an overflowing fugacity at low temperature.
The full state is computed in one species loop, reusing the Bessel evaluations.
The implementation specializes `thermodynamic`; callers do not need to
differentiate Bessel functions with ForwardDiff.

## Charges and units

The charge basis is user-defined. For baryon number, strangeness, and electric
charge, a proton entry could be:

```julia
using FluidumThermodynamics

proton = HadronSpecies(0.938, 2, (1, 0, 1); statistics=:fermion)
antiproton = HadronSpecies(0.938, 2, (-1, 0, -1); statistics=:fermion)
eos = HadronResonanceGas([proton, antiproton])
th = thermodynamic(eos, GrandCanonicalPoint(0.155, 0.02, 0.0, 0.005))
```

This is an illustrative two-species gas, not a complete HRG spectrum. Every entry
must have the same number and ordering of charges. A point with `M` charges uses
the first `M` entries of that basis and sets omitted chemical potentials to zero.
Its state contains derivatives with respect to `(T, μ₁, …, μₘ)`. A zero-charge
point therefore returns only temperature derivatives at all chemical potentials
zero. The point cannot have more charges than the species basis.

Masses use the EOS energy unit, as do temperature and chemical potentials.
For `MeVFm()`, the proton mass is `938.0`. As with other EOS models, the unit
convention is encoded in the EOS and state types, and `unit_scale` is stored
in the EOS. `convert_units(MeVFm(), eos)` converts an existing model without
requiring the caller to rebuild its particle list.

Generic canonical and microcanonical inversion work with an explicit initial
guess. Your charge basis and spectrum must give a nonsingular inversion Jacobian.
