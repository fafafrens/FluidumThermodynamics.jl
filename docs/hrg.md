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
Scaled Bessel functions from [Bessels.jl](https://github.com/JuliaMath/Bessels.jl)
combine the fugacity and mass exponential before evaluation. This avoids
multiplying a vanishing Bessel value by an overflowing fugacity at low temperature.
The full state is computed in one species loop, reusing the Bessel evaluations.
The implementation specializes `thermodynamic`, so callers do not need to
differentiate Bessel functions; the package's `besselkx` wrapper nevertheless
accepts ForwardDiff dual numbers (including nested ones), so `pressure` of an
HRG can be differentiated with ForwardDiff as well.

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

## Reading particle lists

`read_particle_list(file)` builds the species from a FastReso particle list
(`particles.data` format, columns `name mass gamma spin isospin I3 Nq Ns Naq
Nas Nc Nac MC`, masses in GeV), as Fluidum's `readresonancelist` does.
`read_particle_table(file)` returns the raw rows as NamedTuples.

```julia
charm = read_particle_list("particles_charm.data")   # charge Nc + Nac, open charm and J/ψ
hq = HeavyQuark(; particles=charm)                    # bulk EOS + charm hadron gas
```

Each row is one isospin state with degeneracy `2·spin + 1`. FastReso lists
omit antiparticles, so by default (`antiparticles=true`) the degeneracy is
doubled for every row that is not self-conjugate (`Nq = Naq`, `Ns = Nas`,
`Nc = Nac`, `I3 = 0`). The conjugate then carries the same charges, which is
right for the heavy-quark count `charm_count(row) = Nc + Nac` (1 for D mesons
and charm baryons, 2 for J/ψ). For charges that change sign, such as baryon
number, pass `antiparticles=false` and add the conjugates yourself. Keywords:

- `charges = row -> (charm_count(row),)`: the charge tuple of each row, which
  sets the basis.
- `condition`: row filter; by default rows whose charges all vanish are
  dropped, so a full light+charm list yields only the charm hadrons.
- `minmass = 0`, `maxmass = Inf`: open mass window, as in Fluidum.
- `statistics = :boltzmann`: or `:boson`, `:fermion`, or `:quantum` (chosen by
  the spin).

Masses stay in GeV; use an EOS whose energy unit is GeV.
