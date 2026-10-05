# General second-order transport

The package stores constitutive coefficients separately from the equations
that use them. `TransportModel` accepts an optional second-order model, and
`transport` returns its evaluated coefficients in `tr.second_order`.
Existing shear/bulk relaxation times and the charge relaxation matrix remain
in `tr.shear.tau`, `tr.bulk.tau`, and `tr.charge.tau`.

This organization accommodates different second-order truncations and
multi-charge couplings. Examples of such formulations are
[Denicol et al. (2012), including its erratum](https://doi.org/10.1103/PhysRevD.85.114047)
and [Fotakis et al. (2022)](https://arxiv.org/abs/2203.11549).
This package supplies a general coefficient interface, not an implementation
of either paper's complete closure.

## Coefficient structure

```julia
SecondOrderState(:my_equation_convention;
    shear=(delta_pipi=δππ, lambda_piPi=λπΠ),
    bulk=(delta_PiPi=δΠΠ, lambda_Pipi=λΠπ),
    charge=(delta_VV=δVV, ell_VPi=ℓVΠ),
)
```

The three sectors are concrete `NamedTuple` types, grouped by the equation
receiving a term. Thus a bulk-to-shear coupling belongs to `shear`, and a
shear-to-bulk coupling belongs to `bulk`.
Coefficient names are extensible. There are no built-in assumptions about
symmetry, sign, or relations between different coefficients.

`basis` is an explicit symbolic label for a convention documented by the
model author. The label itself does not enforce a convention. Document the
hydrodynamic frame, dissipative variables, signs, normalization, units, and
the term multiplying each coefficient. In particular, distinguish a
coefficient in a relaxation equation from its value after division by a
relaxation time.

An omitted coefficient is unspecified. Supply an explicit zero if your
closure sets it to zero. The default `NoSecondOrderTransport()` returns
`nothing`, meaning no additional coefficient set was supplied; it does
not remove relaxation times from the leading sectors.

## Charge indices and shape checks

Arrays contain **charge indices only**. They are not spacetime tensors.
Every array axis must be `1:N`, where `N = length(x.μ)`, including arrays
appearing in shear or bulk coefficients. Charge ordering follows the EOS.

| Example entry | Shape | Possible role, to be defined by the basis |
|---|---|---|
| `shear.lambda_piPi` | Scalar | Bulk pressure entering the shear equation |
| `bulk.lambda_Pipi` | Scalar | Shear entering the bulk equation |
| `shear.ell_piV` | N-vector | Sum over currents entering the shear equation |
| `charge.ell_VPi` | N-vector | Bulk gradient entering each current equation |
| `charge.delta_VV` | N×N matrix | Coupled diffusion-expansion term |
| `charge.nonlinear_VVV` | N×N×N array | Output charge and two input charge indices |

For charge coefficients, the first axis denotes the output current and the
basis defines the remaining axes. Even one-charge coefficients use an
explicit vector/matrix rather than a scalar. For zero charges, omit the
charge sector or use empty arrays of the required rank.

The container accepts arbitrary numeric array ranks. It checks axis sizes,
not whether a coefficient's rank, units, tensor symmetries, or physical
meaning are correct for a specific basis. Those belong to the model.
Shape checks run when `transport` evaluates the model.

## Constant and state-dependent models

```julia
constant = ConstantSecondOrderTransport(:my_equation_convention;
    shear=(delta_pipi=0.2,),
    bulk=(lambda_Pipi=0.1,),
)

dependent = FunctionalSecondOrderTransport() do x,th,leading
    SecondOrderState(:my_equation_convention;
        shear=(delta_pipi=1.1*leading.shear.tau,),
        bulk=(delta_PiPi=0.9*leading.bulk.tau,),
        charge=(delta_VV=1.2*leading.charge.tau,),
    )
end

model = TransportModel(shear_model,bulk_model,charge_model;
                       second_order=dependent)
th,tr = transport(FluidModel(eos,model),x)
δ = tr.second_order.shear.delta_pipi
```

The numbers above are illustrative inputs. The callback receives the
grand-canonical point, its thermodynamic state, and the evaluated leading
transport sectors. It can therefore depend on temperature, chemical
potentials, susceptibilities, viscosities, conductivity, and relaxation times
without evaluating them again. `leading.second_order` is `nothing`.

For reusable physical models, define a subtype of
`AbstractSecondOrderTransport` and specialize
`FluidumThermodynamics.second_order_state(model,x,th,leading)`.
Return a `SecondOrderState`; `transport` performs the same shape validation
as for the built-in models.

The containers retain array references. StaticArrays are convenient for small
fixed charge counts; copy mutable arrays when independent ownership is needed.

## Example equation convention

The runnable [example](../examples/second_order.jl) names its convention
`:example_landau_relaxation`. It defines the supplied coefficients through
the following illustrative truncated relaxation equations:

```text
τπ Dπ^{<μν>} + π^{μν}
    = 2η σ^{μν}
      - δππ π^{μν} θ
      - τππ π_λ^{<μ} σ^{ν>λ}
      + λπΠ Π σ^{μν}

τΠ DΠ + Π
    = -ζ θ
      - δΠΠ Π θ
      + λΠπ π^{μν} σ_{μν}

Σ_b τ[a,b] Δ^μ_ν D V_b^ν + V_a^μ
    = Σ_b κ[a,b] ∇^μ(μ_b/T)
      - Σ_b δVV[a,b] V_b^μ θ
```

Here `D = u·∂`, `θ = ∂·u`, and angle brackets denote symmetric,
traceless projection orthogonal to the flow. This example uses the Landau
frame and metric signature `(+,-,-,-)`, with
`Δ^{μν} = g^{μν} - u^μ u^ν`, `∇^μ = Δ^{μν}∂_ν`, and
`σ^{μν} = ∇^{<μ}u^{ν>}`.
The example includes only the displayed terms and uses supplied
phenomenological ratios; it does not claim a complete kinetic-theory closure.
The `tau_pipi` coefficient is separate from the relaxation time
`tr.shear.tau`.

Run it from the package directory:

```sh
julia --project=. examples/second_order.jl
```

The example evaluates coefficients and demonstrates one charge-matrix
contraction. Evaluating spacetime gradients and projections or evolving the
relaxation equations remains the responsibility of a hydrodynamic solver.
