# FluidumThermodynamics.jl

Experimental Julia package for multi-charge thermodynamics and transport
coefficients, extracted from [Fluidum.jl](https://github.com/fafafrens/Fluidum.jl/pull/52).
Runs independently of Fluidum. Requires Julia 1.10 or newer; the API may change.

## Install

The package is unregistered. With access to this private repository:

```julia
using Pkg
Pkg.add(url="https://github.com/fafafrens/FluidumThermodynamics.jl")
```

## Quick start

```julia
using FluidumThermodynamics

eos = IdealQCD()
x = GrandCanonicalPoint(0.2, 0.03)  # T, μ in GeV
th = thermodynamic(eos, x)

p = th.pressure
s = entropy_density(th)
n = charge_densities(th)
ε = energy_density(x, th)
```

`ThermodynamicState` contains pressure, its gradient, and its Hessian with
respect to `(T, μ...)`. Canonical `(T, n...)` and microcanonical `(ε, n...)`
coordinates are also supported.

Included models: `IdealQCD`, the zero-charge `FluidumEOS` fit, and
`PolynomialMultiChargeEOS`. Custom models can supply pressure for automatic
differentiation or implement analytic derivatives.

Choose units on the EOS: `NaturalGeV()` (default, pressure in GeV⁴),
`GeVFm()`, or `MeVFm()`. Results inherit the EOS unit convention;
`convert_units` explicitly converts models and states.

```julia
eos = FluidumEOS(units=MeVFm())
th = thermodynamic(eos, GrandCanonicalPoint(200.0))  # pressure in MeV/fm³
```

Transport models cover shear, bulk, charge diffusion, and named second-order
coefficients.

## More

- [User guide](docs/guide.md): units, custom EOS models, inversion, and transport.
- [Second-order transport](docs/second_order.md).
- [Examples](examples) and [benchmarks](benchmark/README.md).

Run tests with `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`.

MIT licensed; retains the original Fluidum copyright.
