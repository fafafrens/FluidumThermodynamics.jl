# FluidumThermodynamics.jl

An experimental Julia package for multi-charge thermodynamics and constitutive
transport models, extracted from [Fluidum.jl PR #52](https://github.com/fafafrens/Fluidum.jl/pull/52).
It runs independently of Fluidum and has no hydrodynamic solver dependency.
The API is experimental and may change before a stable release.

## Local installation

Requires Julia 1.10 or newer. From another Julia environment:

```julia
using Pkg
Pkg.develop(path="/path/to/FluidumThermodynamics.jl")
using FluidumThermodynamics
```

To instantiate and test this checkout:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. examples/multicharge.jl
```

The package is unregistered. Its dependencies are ForwardDiff, StaticArrays, SpecialFunctions,
and the LinearAlgebra standard library.

## Thermodynamics

```julia
using FluidumThermodynamics
using StaticArrays

eos = IdealQCD(3,2)
x = GrandCanonicalPoint(0.2,0.03)
th = thermodynamic(eos,x)

p = th.pressure
s = entropy_density(th)
n = charge_densities(th)
χ = susceptibilities(th)
ε = energy_density(x,th)
w = enthalpy(x,th)              # ε + p
cs2 = speed_of_sound_squared(x,th)

can = canonical(x,th)          # (T,n)
mic = microcanonical(x,th)     # (ε,n)
from_can = grandcanonical(eos,can,SVector(0.0))
from_mic = grandcanonical(eos,mic,GrandCanonicalPoint(0.18,0.0))
```

- `GrandCanonicalPoint(T, μ...)` stores coordinates `(T,μ₁,…,μₙ)`.
- `CanonicalPoint(T, n...)` stores `(T,n₁,…,nₙ)`.
- `MicrocanonicalPoint(ε, n...)` stores `(ε,n₁,…,nₙ)`.
- `ThermodynamicState` stores pressure, its gradient `(s,n)`, and its Hessian
  in grand-canonical coordinates. Its Hessian is not transformed into another
  ensemble when converting coordinates.

Each point also accepts an `SVector` of charges. A single argument constructs
a zero-charge point.

Derived quantities reuse the stored gradient and Hessian, with no further EOS
evaluation:

| Function | Quantity |
|---|---|
| `enthalpy(x,th)` | `ε + p = T s + μ⋅n` |
| `trace_anomaly(x,th)` | `ε - 3p` |
| `pressure_derivatives(x,th)` | `(∂p/∂ε at fixed n, ∂p/∂nᵢ at fixed ε)` |
| `speed_of_sound_squared(x,th)` | `c_s²` at fixed entropy per charge |

Each also accepts `(eos,x)` and evaluates the EOS once. The speed of sound is
`c_s² = ∂p/∂ε + Σᵢ nᵢ/(ε+p) ∂p/∂nᵢ`, which reduces to `s/(T ∂s/∂T)` without
charges and to `1/3` for `IdealQCD`.

Plain numbers are shorthand for a grand-canonical point, in the EOS unit
convention:

```julia
thermodynamic(eos,0.2,0.03)          # thermodynamic(eos,GrandCanonicalPoint(0.2,0.03))
thermodynamic(eos,0.2,SVector(0.03)) # same point
thermodynamic(eos,0.2)               # zero-charge point
```

Numbers are never read as canonical or microcanonical coordinates; build a
`CanonicalPoint` or `MicrocanonicalPoint` for those. Energy density is the Legendre transform
`ε = -p + T*s + dot(μ,n)`.

Included EOS implementations:

| Model | Charges | Convention |
|---|---|---|
| `IdealQCD(Nc=3,Nf=2; units=NaturalGeV())` | Zero or one | Preserves Fluidum's physical normalization; evaluates in the chosen convention |
| `FluiduMEoS(; units=NaturalGeV())` (also `FluidumEOS`) | Zero | Original Fluidum temperature-dependent fit with analytic first and second derivatives |
| `HadronResonanceGas(particles; units=NaturalGeV(), terms=16)` | Any fixed number | Supplied resonance spectrum, Boltzmann or truncated Bose/Fermi statistics, analytic derivatives |
| `PolynomialMultiChargeEOS(a,χ; units=NaturalGeV())` | Any fixed number | Toy pressure `a*T^4 + T^2*dot(μ,χ*μ)/2`; supplied coefficients use the chosen convention |
| `TabulatedEOS(T,p,s,dsdT; units=NaturalGeV())` | Zero | Quintic Hermite table of `p/T⁴`; thermodynamically consistent and guess-free inversion |

The original `IdealQCD` normalization and conversion `ℏc = 0.1973261 GeV fm`
are preserved. Its single-charge convention is not generalized automatically
to other charge bases. `FluiduMEoS` preserves the original fit coefficients
and its 154/100 MeV temperature scales. Evaluate it with a zero-charge point:

```julia
eos = FluidumEOS(units=MeVFm())
th = thermodynamic(eos,GrandCanonicalPoint(200.0))
```

The fit has no finite-chemical-potential extension.

### Hadron resonance gas

Supply masses in the EOS energy unit, degeneracies, and charges in your chosen
basis. Include antiparticles explicitly. This small example uses baryon number:

```julia
particles = [
    HadronSpecies(0.938, 2, (1,); statistics=:fermion),
    HadronSpecies(0.938, 2, (-1,); statistics=:fermion),
    HadronSpecies(0.140, 1, (0,); statistics=:boson),
]
eos = HadronResonanceGas(particles; terms=16)
th = thermodynamic(eos, GrandCanonicalPoint(0.155, 0.02))
```

The default species statistics are `:boltzmann` (one term). For quantum species,
`terms` controls the fugacity expansion; increase it to check convergence.
See the [HRG guide](docs/hrg.md) for the formula, multi-charge usage, and limits.
No complete particle catalog is bundled.

EOS models support addition, subtraction, negation, scalar multiplication,
and scalar division. Composition combines evaluated pressure, gradient,
and Hessian, preserving specialized `thermodynamic` implementations.

### Tabulated EOS

`TabulatedEOS` interpolates a zero-charge table, for example lattice QCD data.
It interpolates the single function `p/T⁴` with quintic Hermite polynomials
that match `p`, `s`, and `∂s/∂T` at every node. Entropy and `∂s/∂T` are
derivatives of that one interpolant, so the table is thermodynamically
consistent, and `p` is twice continuously differentiable.

```julia
table = TabulatedEOS(T, p, s, dsdT)              # columns in the EOS convention
table = TabulatedEOS(T, p, s)                    # dsdT estimated from s
table = TabulatedEOS(FluidumEOS(), range(0.08, 0.6, length=200))  # sample a model
th = thermodynamic(table, 0.2)
x = grandcanonical(table, MicrocanonicalPoint(ε)) # no initial guess needed
```

Temperatures must be positive and strictly increasing, but need not be
uniform. When a lattice table lists the speed of sound instead of `∂s/∂T`, use
`dsdT = s ./ (T .* cs2)`. Without `dsdT`, it is estimated by fourth-order finite
differences of `s/T³`, which is less accurate. The table is not extrapolated;
evaluation or inversion outside its range throws a `DomainError`.

Sampling `FluidumEOS` on 100 nodes from 0.08 to 0.6 GeV reproduces pressure to
about `4e-9`, entropy to `5e-8`, and `∂s/∂T` to `2e-6`, relative. The error falls
as the sixth power of the node spacing. A conformal `p ∝ T⁴` is exact.
Because energy density increases with temperature, `ε -> T` inversion brackets
the root in one table cell and needs no initial guess.

## Unit conventions

Choose units once, when constructing the EOS. Inputs remain plain numbers,
and every thermodynamic result inherits the EOS convention in its type.
`NaturalGeV()` is the default: pressure is in GeV⁴ and the stored
`unit_scale` is exactly 1. Temperature is expressed in GeV with
`ℏ = c = k_B = 1`. Select `GeVFm()` or `MeVFm()` explicitly for pressure
in GeV/fm³ or MeV/fm³.

```julia
eos = IdealQCD(units=MeVFm())
x = GrandCanonicalPoint(200.0,30.0)  # MeV
th = thermodynamic(eos,x)

unit_system(eos) == unit_system(th) == MeVFm()  # true
th.pressure  # MeV/fm³
```

| Quantity | NaturalGeV (default) | GeVFm | MeVFm |
|---|---|---|---|
| Temperature, chemical potentials | GeV | GeV | MeV |
| Pressure, energy density | GeV⁴ | GeV/fm³ | MeV/fm³ |
| Entropy and charge densities | GeV³ | fm⁻³ | fm⁻³ |
| Pressure Hessian | GeV² | GeV⁻¹ fm⁻³ | MeV⁻¹ fm⁻³ |

To change conventions explicitly:

```julia
eos_gev = convert_units(GeVFm(),eos)
x_gev = convert_units(GeVFm(),MeVFm(),x)
th_gev = thermodynamic(eos_gev,x_gev)
# Equivalently, convert an already evaluated result:
converted_state = convert_units(GeVFm(),th)
```

Conversion rescales coordinates, pressure, gradient, and Hessian with their
respective factors; it returns a new object. An EOS conversion wraps the
original evaluation, preserving specialized derivatives and tabulated models.
Coordinate points have no unit tag, so their source convention is explicit
when converting. Canonical and microcanonical points can be converted in the
same way. Inversion targets and guesses use the EOS convention.

EOS addition/subtraction requires matching unit-system types. Scalar
multiplication and division preserve the convention. No unit validation is
performed inside the numerical evaluation.

`PolynomialMultiChargeEOS(a,χ; units=...)` interprets the supplied coefficients
in that convention (pressure/energy⁴). To preserve an existing model's physical
predictions when changing units, use `convert_units`.
Transport coefficients remain supplied in their model's convention; these
conversion methods cover EOS models and thermodynamic data.

Custom unit systems are zero-field subtypes of `AbstractUnitSystem`, defining
`energy_scale` and `length_scale` as the sizes of their energy and length units
in GeV and fm, respectively.

## Implementing an EOS

Every EOS must store a `unit_scale` field initialized from its unit convention.
`unit_scale(eos)` reads this cached value; the conversion formula remains in
`units.jl`. Composed and converted EOS models carry the field too.
Define an `AbstractEquationOfState` subtype and its pressure:

```julia
struct MyEOS{U,T,S} <: AbstractEquationOfState{U}
    a::T
    unit_scale::S
end

function MyEOS(a; units::U=NaturalGeV()) where {U<:AbstractUnitSystem}
    scale = unit_scale(units)
    MyEOS{U,typeof(a),typeof(scale)}(a,scale)
end

FluidumThermodynamics.pressure(eos::MyEOS,x::GrandCanonicalPoint) =
    unit_scale(eos)*eos.a*x.T^4
```

Here `a` is dimensionless. Apply the factor when converting an energy⁴
expression to pressure. Models whose coefficients or table values already
include this conversion (such as `PolynomialMultiChargeEOS`) must not apply
it a second time.

The generic `thermodynamic(eos,x)` fallback evaluates pressure, gradient,
and Hessian with ForwardDiff. Pressure must accept ForwardDiff dual numbers.
An analytic or tabulated EOS can specialize `FluidumThermodynamics.thermodynamic`
and return `ThermodynamicState(eos,p,g,H)`, with `g::SVector` and `H::SMatrix`.
This constructor inherits the unit type from the EOS. Custom specializations
are responsible for using this convention.
The derivatives must be with respect to `(T,μ...)`, in that order.

## Inversion

`IdealQCD` also accepts canonical and microcanonical points directly:

```julia
eos = IdealQCD()
x = GrandCanonicalPoint(0.2,0.03)
can = canonical(eos,x)
mic = microcanonical(eos,x)
th_can = thermodynamic(eos,can)
th_mic = thermodynamic(eos,mic)
```

For canonical input, `IdealQCD` analytically solves
`n = b*μ + c*μ³`, using the real hyperbolic cubic solution
([NIST DLMF](https://dlmf.nist.gov/4.43)).
For microcanonical input at zero charge density, it uses
`T = (8ε/(unit_scale(eos)*a1))^(1/4)`.
At nonzero density it uses the existing numerical inverse with an automatically
generated guess; an explicit `guess` and solver keywords can be supplied.
`grandcanonical(eos,can)` and `grandcanonical(eos,mic)` expose the recovered
coordinates. The evaluated state always stores derivatives with respect to
`(T,μ...)`; these overloads do not change the derivative basis.

Canonical inversion uses the susceptibility matrix. Microcanonical inversion
uses the pressure Hessian to form the Jacobian of `(ε,n)`.
Both use damped Newton steps and component-wise tolerances:
`abs(residual[i]) <= atol + rtol*abs(target[i])`.

Defaults are `atol=1e-12`, `rtol=1e-10`, `maxiter=30`, and
`maxbacktracks=30`. Choose tolerances appropriate to your model's units.
The optional `domain(x)` predicate constrains candidate coordinates; by default
temperature must be positive and all coordinates finite. For example:

```julia
result = grandcanonical(eos,mic,GrandCanonicalPoint(0.18,0.0);
    domain=x -> 0.1 <= x.T <= 0.5 && abs(x.μ[1]) <= 0.3)
```

An invalid initial guess or invalid solver options raise `ArgumentError`.
Singular steps, unsuccessful damping, or exhausted iterations raise
`InversionError`, with a reason, iteration count, and residual norm.
`DomainError` from an EOS evaluation rejects a trial step.
Inversion is local; a guess selects a branch and convergence is not guaranteed
across phase transitions or singular susceptibilities.
`invert` is an alias for `grandcanonical`.

## Transport

```julia
model = TransportModel(
    ConstantEtaOverS(0.16,5.0),
    ConstantZetaOverS(0.03,10.0),
    ZeroChargeTransport(),
)
th,tr = transport(FluidModel(eos,model),x)
η,τπ = tr.shear.eta,tr.shear.tau
ζ,τΠ = tr.bulk.zeta,tr.bulk.tau
```

`transport(fluid,T,μ...)` is shorthand for
`transport(fluid,GrandCanonicalPoint(T,μ...))`, for example
`th,tr = transport(FluidModel(eos,model),0.2,0.03)`.

`FluidModel` is an EOS/transport container; it does not evolve a fluid.
The constant shear and bulk models use
`η = (η/s)*s`, `ζ = (ζ/s)*s`, and
`τ = viscosity/[Ctau*(ε+p)]`. The caller supplies physically appropriate
coefficients and states with nonzero enthalpy.

`ConstantChargeTransport(κ,τ)` accepts matrices whose dimensions match the
number of charges, including off-diagonal couplings.
`ZeroShearTransport`, `ZeroBulkTransport`, and `ZeroChargeTransport`
disable their respective sectors.

Custom transport models subtype `AbstractShearTransport`,
`AbstractBulkTransport`, or `AbstractChargeTransport` and specialize
`FluidumThermodynamics.shear_state(model,x,th)`,
`bulk_state(model,x,th)`, or `charge_state(model,x,th)`, respectively.
Return `ShearState`, `BulkState`, or `ChargeState`.
`transport(model,x,th)` reuses an already evaluated thermodynamic state.

### General second-order coefficients

`TransportModel` also accepts an optional second-order model:

```julia
second = FunctionalSecondOrderTransport() do x,th,leading
    SecondOrderState(:my_relaxation_convention;
        shear=(delta_pipi=1.1*leading.shear.tau,),
        bulk=(delta_PiPi=0.9*leading.bulk.tau,),
        charge=(delta_VV=1.2*leading.charge.tau,),
    )
end
model = TransportModel(
    ConstantEtaOverS(0.16,5.0),
    ConstantZetaOverS(0.03,10.0),
    ZeroChargeTransport();
    second_order=second,
)
th,tr = transport(FluidModel(eos,model),x)
δππ = tr.second_order.shear.delta_pipi
```

The ratios here are demonstration inputs. `SecondOrderState` groups named
coefficients by the shear, bulk, or charge equation they enter, including
cross-couplings and arbitrary-rank arrays over charge indices.
`ConstantSecondOrderTransport` supplies fixed coefficients; custom models
extend `AbstractSecondOrderTransport` and `second_order_state`.
All array axes must match the number of charges.

Coefficient meanings, units, and signs belong to the explicitly named
convention. Missing coefficients are unspecified, not implicit zeros.
The default `NoSecondOrderTransport()` yields `tr.second_order === nothing`;
relaxation times remain in their existing sectors.
See the [second-order guide](docs/second_order.md) for the shape rules,
example equations, and extension interface, and run
`julia --project=. examples/second_order.jl` for a two-charge example.

## Provenance and validation

Reproducible FluidumEOS timings, comparison with the original Fluidum kernel,
and high-precision derivative checks are in the [benchmark guide](benchmark/README.md).

The starting prototype, original smoke tests, and EOS references come from
Fluidum.jl commit `4fdd25fc311d64619d8b2f2af44054775dbec50f`.
The MIT license and original copyright are retained.
The reference snapshot in `test/fixtures/` is used only for independent
comparisons; installing this package does not load Fluidum.

Tests cover analytic and AD derivatives, zero/multiple charges, ensemble round
trips, specialized EOS composition, inversion failure paths, transport, and
agreement with the original IdealQCD and FluiduMEoS implementations. CI is configured for
Julia 1.10 and the latest stable Julia release.
