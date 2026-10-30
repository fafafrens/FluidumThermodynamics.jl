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

`read_particle_list(file)` builds the species from a FastReso particle list, as
Fluidum's `readresonancelist` does; `read_particle_table(file)` returns the raw
rows as NamedTuples. Two formats are recognized (`format=:auto` decides from
the first row, or pass `:canonical` / `:therminator`):

- **canonical**, the FastReso2 tables `particles_*.dat`:
  `id name mass width degeneracy B S C Bot I Q [aliases...]`. The degeneracy is
  given and antiparticles are rows of their own (negative PDG id); `C` is the
  net charm.
- **THERMINATOR**, the 13-column `particles.data` layout:
  `name mass gamma spin isospin I3 Nq Ns Naq Nas Nc Nac MC`, one isospin state
  per row with degeneracy `2·spin + 1` and no antiparticles.

```julia
charm = read_particle_list("particles_charm.dat")     # c + c̄: D, D̄ → 1, J/ψ → 2
hq = HeavyQuark(; particles=charm)                     # bulk EOS + charm hadron gas
open_charm = read_particle_list("particles_charm.dat"; net=true)   # D → +1, D̄ → -1
```

The default charge is `charm_count(row; net)`. With `net=false` (default) it is
the heavy-quark count `c + c̄`: `Nc + Nac` for THERMINATOR rows, and for
canonical rows the charm digits of the PDG id `±(n nr nL nq1 nq2 nq3 nJ)`.
Quark-model states without a PDG id (e.g. `5000440`, an N*) have numbers whose
digits are not flavours, so the digits are trusted only where they agree with
`C`: `|C|` for open charm, 2 for a meson with `C = 0` whose two quark digits are
both charm (J/ψ, χc, ψ(4040) = 9000443, …), 0 otherwise. With `net=true` the
charge is the net charm, `C` or `Nc - Nac`, and charmonia drop out.

THERMINATOR lists omit antiparticles, so for them `antiparticles=true` is the
default: the degeneracy is doubled for every row that is not self-conjugate
(`Nq = Naq`, `Ns = Nas`, `Nc = Nac`, `I3 = 0`). The conjugate then carries the
same charges, which is right for `c + c̄`; for charges that change sign, such
as the net charm, pass `antiparticles=false` and add the conjugates yourself.
Canonical lists already contain the antiparticles, and `antiparticles=true` is
rejected for them. Other keywords:

- `charges = row -> (charm_count(row; net),)`: the charge tuple of each row,
  which sets the basis.
- `condition`: row filter; by default rows whose charges all vanish are
  dropped, so a full light+charm list yields only the charm hadrons.
- `minmass = 0`, `maxmass = Inf`: open mass window, as in Fluidum.
- `statistics = :boltzmann`: or `:boson`, `:fermion`, or `:quantum`
  (Fermi-Dirac for baryons, Bose-Einstein otherwise).

Masses stay in GeV; use an EOS whose energy unit is GeV.
