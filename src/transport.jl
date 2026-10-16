
abstract type AbstractShearTransport end
abstract type AbstractBulkTransport end
abstract type AbstractChargeTransport end

struct ZeroShearTransport <: AbstractShearTransport end
struct ZeroBulkTransport <: AbstractBulkTransport end
struct ZeroChargeTransport <: AbstractChargeTransport end

"""Simple η/s model with τπ = η/[Cτ(ε+p)]."""
struct ConstantEtaOverS{T} <: AbstractShearTransport
    eta_over_s::T
    Ctau::T
end

"""Simple ζ/s model with τΠ = ζ/[Cτ(ε+p)]."""
struct ConstantZetaOverS{T} <: AbstractBulkTransport
    zeta_over_s::T
    Ctau::T
end

"""
    ConstantChargeTransport(kappa, tau)

Coupled charge-transport model. Conductivity and relaxation time are matrices,
so off-diagonal transport among conserved charges is supported directly.
Plain matrices are copied into static matrices, so evaluation does not
allocate; later changes to the original matrices do not affect the model.

The entries are plain numbers in the EOS convention (κ in length⁻², τ in
length), fixed independently of the state. Since κ carries dimension
energy², a fixed value is a unit-dependent statement whose diffusion front
`κ/(τ ∂n/∂(μ/T))` changes with temperature; prefer `ConstantKappaOverT2` or
`ConstantDsT`, whose inputs are dimensionless.
"""
struct ConstantChargeTransport{K,R} <: AbstractChargeTransport
    kappa::K
    tau::R
    function ConstantChargeTransport(kappa, tau)
        k, t = _static_matrix(kappa), _static_matrix(tau)
        new{typeof(k),typeof(t)}(k,t)
    end
end

"""
    CharmTransport(eos, DsT, charge=1)

Free (Boltzmann) charm diffusion with spatial diffusion coefficient `D_s`.
`eos` is a single-species heavy-quark EOS such as `HeavyQuark()`; its species
mass is the charm mass, in the EOS energy unit. `DsT` is the dimensionless
`D_s*T`, either a number or a callable `T -> D_s*T`. With `n` the charm
density of `eos` evaluated at `(T, μ[charge])` and `z = mass/T`,

    κ = (D_s T)/T * n
    τ = (D_s T) * z^3/(48T) * (2K₁(z) - 3K₃(z) + K₅(z))/K₂(z)

so τ → D_s*mass/T for heavy quarks. Only the `[charge,charge]` entries are
nonzero. Coefficients use ħ = c = 1 in the EOS energy convention.
"""
struct CharmTransport{E<:AbstractEquationOfState,D} <: AbstractChargeTransport
    eos::E
    DsT::D
    charge::Int
    function CharmTransport(eos::AbstractEquationOfState, DsT, charge::Integer=1)
        charge >= 1 || throw(ArgumentError("charm charge index must be positive"))
        heavy_quark_mass(eos)  # validates that the EOS has a single species
        new{typeof(eos),typeof(DsT)}(eos,DsT,Int(charge))
    end
end

@inline _diffusion_DsT(DsT::Real, T) = DsT
@inline _diffusion_DsT(DsT, T) = DsT(T)

"""
    ConstantKappaOverT2(kappa_over_T2, Ctau)

Diagonal charge transport with a dimensionless conductivity ratio,

    κ = (κ/T²)·T²,        τ_ν = κ / [Ctau · T ∂n/∂μ],

for every charge. `κ` multiplies `∇(μ/T)`, so `T ∂n/∂μ = ∂n/∂(μ/T)` is the
susceptibility in the fugacity variable and the diffusion front speed

    v_D² = κ / (τ_ν ∂n/∂(μ/T)) = Ctau

is the same number at every state: `Ctau < 1` is causal by construction, the
charge analogue of the transverse shear front `η/(τ_π w) = Ctau` of
`ConstantEtaOverS`. The ratio `κ/T²` is the natural constant of a conformal
plasma. With several charges `κ` is `(κ/T²)·T²·I` and `τ_ν` is the matrix
`κ·(Ctau·T·χ)⁻¹` with `χ` the susceptibility matrix.
"""
struct ConstantKappaOverT2{T} <: AbstractChargeTransport
    kappa_over_T2::T
    Ctau::T
end

ConstantKappaOverT2(kappa_over_T2, Ctau) =
    ConstantKappaOverT2(promote(kappa_over_T2,Ctau)...)

"""
    ConstantDsT(DsT, Ctau)

Diagonal charge transport in the Einstein form of a heavy-quark or hadron
diffusion coefficient, with the dimensionless spatial diffusion constant
`Dₛ T` fixed:

    κᵢ = Dₛ·nᵢ = (DₛT)·nᵢ/T,        τ_ν = κ / [Ctau · T ∂n/∂μ].

This is the convention of Fluidum's `ConstDiffusion(DsT, mass)`, with the
relaxation time set by the front speed `v_D² = Ctau` instead of the
kinetic-theory Bessel-function formula. `κ` vanishes with the density, so at
`μ = 0` of a charge-symmetric EOS this model gives no diffusion; it is meant
for an abundance of a conserved species, such as heavy flavour.
"""
struct ConstantDsT{T} <: AbstractChargeTransport
    DsT::T
    Ctau::T
end

ConstantDsT(DsT, Ctau) = ConstantDsT(promote(DsT,Ctau)...)

"""
    TransportModel(shear, bulk, charge; second_order=NoSecondOrderTransport())

Combine the shear, bulk, and charge sectors with optional additional
second-order coefficients. The fourth model can also be passed positionally.
"""
struct TransportModel{S<:AbstractShearTransport,B<:AbstractBulkTransport,C<:AbstractChargeTransport,O<:AbstractSecondOrderTransport}
    shear::S
    bulk::B
    charge::C
    second_order::O
end

TransportModel(shear::AbstractShearTransport, bulk::AbstractBulkTransport,
               charge::AbstractChargeTransport;
               second_order::AbstractSecondOrderTransport=NoSecondOrderTransport()) =
    TransportModel(shear,bulk,charge,second_order)

struct FluidModel{E<:AbstractEquationOfState,T<:TransportModel}
    eos::E
    transport::T
end

struct ShearState{T}
    eta::T
    tau::T
end

struct BulkState{T}
    zeta::T
    tau::T
end

struct ChargeState{K,R}
    kappa::K
    tau::R
end

"""
    TransportState(shear, bulk, charge[, second_order])

Evaluated transport coefficients. `second_order` is a `SecondOrderState`, or
`nothing` when no additional coefficients are supplied. The three-argument
constructor defaults to `nothing`.
"""
struct TransportState{S,B,C,O<:Union{Nothing,SecondOrderState}}
    shear::S
    bulk::B
    charge::C
    second_order::O
end

TransportState(shear,bulk,charge) = TransportState(shear,bulk,charge,nothing)

@inline function shear_state(::ZeroShearTransport, x, th)
    z = zero(th.pressure)
    ShearState(z,z)
end

@inline function shear_state(model::ConstantEtaOverS, x, th)
    eta = model.eta_over_s*entropy_density(th)
    tau = eta/(model.Ctau*enthalpy(x,th))
    ShearState(eta,tau)
end

@inline function bulk_state(::ZeroBulkTransport, x, th)
    z = zero(th.pressure)
    BulkState(z,z)
end

@inline function bulk_state(model::ConstantZetaOverS, x, th)
    zeta = model.zeta_over_s*entropy_density(th)
    tau = zeta/(model.Ctau*enthalpy(x,th))
    BulkState(zeta,tau)
end

@inline function charge_state(::ZeroChargeTransport, x::GrandCanonicalPoint{T,N}, th) where {T,N}
    z = zero(th.pressure)
    Z = zeros(SMatrix{N,N,typeof(z)})
    ChargeState(Z,Z)
end

@inline function charge_state(model::ConstantChargeTransport, x::GrandCanonicalPoint{T,N}, th) where {T,N}
    size(model.kappa) == (N,N) && size(model.tau) == (N,N) ||
        throw(DimensionMismatch("charge transport matrices must match the number of charges"))
    ChargeState(model.kappa,model.tau)
end

@inline function charge_state(model::CharmTransport, x::GrandCanonicalPoint{T,N}, th) where {T,N}
    c = model.charge
    c <= N || throw(DimensionMismatch("charm charge index must not exceed the number of charges"))
    temperature = x.T
    DsT = _diffusion_DsT(model.DsT,temperature)
    z = heavy_quark_mass(model.eos)/temperature
    # Scaled Bessel functions: the common e^z factor cancels in the ratio.
    b1, b2 = besselkx(1,z), besselkx(2,z)
    b3 = muladd(4/z,b2,b1)
    b4 = muladd(6/z,b3,b2)
    b5 = muladd(8/z,b4,b3)
    # The charm density comes from the heavy-quark EOS, not from the fluid's `th`.
    hq = thermodynamic(model.eos, GrandCanonicalPoint(temperature, SVector(x.μ[c])))
    kappa = DsT/temperature*charge_density(hq,1)
    tau = DsT*z^3/(48*temperature)*(2*b1-3*b3+b5)/b2
    k, t = promote(kappa,tau)
    i = (c-1)*N+c
    K = SMatrix{N,N}(ntuple(j -> j == i ? k : zero(k), Val(N*N)))
    R = SMatrix{N,N}(ntuple(j -> j == i ? t : zero(t), Val(N*N)))
    ChargeState(K,R)
end

# One length unit of the convention in GeV⁻¹. A quantity of dimension energy^a·length^b
# computed in GeV is divided by energy_scale^a·(length unit)^b to land in the convention.
@inline _length_unit_GeVinv(units::AbstractUnitSystem) = length_scale(units)/_hbarc_GeV_fm

# κ = k̂·T² in the convention of `units`: T² (energy²) converted to length⁻².
@inline _kappa_from_T2(k̂, T, units::AbstractUnitSystem) =
    k̂*T^2*(energy_scale(units)*_length_unit_GeVinv(units))^2

# κ = DₛT·n/T in the convention of `units`: n/T (length⁻³·energy⁻¹) converted to length⁻².
@inline _kappa_from_DsT(DsT, n, T, units::AbstractUnitSystem) =
    DsT*n/(T*energy_scale(units)*_length_unit_GeVinv(units))

# Relaxation-time matrix τ_ν = κ·(Ctau·T·χ)⁻¹ from a diagonal conductivity, so that the front
# speed κ/(τ_ν·T·χ) is Ctau in every charge direction. T·χ is ∂n/∂(μ/T) at fixed T.
@inline function _diagonal_charge_state(kappas::SVector{N,K}, Ctau, x::GrandCanonicalPoint, th) where {N,K}
    kappa = SMatrix{N,N,K}(ntuple(k -> (k-1) % N == (k-1) ÷ N ? kappas[(k-1) % N + 1] : zero(K), N*N))
    N == 0 && return ChargeState(kappa,kappa)
    tau = kappa*inv(Ctau*x.T*susceptibilities(th))
    ChargeState(kappa,tau)
end

@inline function charge_state(model::ConstantKappaOverT2, x::GrandCanonicalPoint{T,N}, th) where {T,N}
    κ = _kappa_from_T2(model.kappa_over_T2,x.T,unit_system(th))
    _diagonal_charge_state(SVector{N,typeof(κ)}(ntuple(_ -> κ, N)),model.Ctau,x,th)
end

@inline function charge_state(model::ConstantDsT, x::GrandCanonicalPoint{T,N}, th) where {T,N}
    n = charge_densities(th)
    units = unit_system(th)
    kappas = SVector{N}(ntuple(i -> _kappa_from_DsT(model.DsT,n[i],x.T,units), N))
    _diagonal_charge_state(kappas,model.Ctau,x,th)
end

@inline function transport(model::TransportModel, x::GrandCanonicalPoint, th::ThermodynamicState)
    leading = TransportState(
        shear_state(model.shear,x,th),
        bulk_state(model.bulk,x,th),
        charge_state(model.charge,x,th),
    )
    second_order = _validate_second_order(second_order_state(model.second_order,x,th,leading),x)
    TransportState(leading.shear,leading.bulk,leading.charge,second_order)
end

@inline function transport(fluid::FluidModel, x::GrandCanonicalPoint)
    th = thermodynamic(fluid.eos,x)
    th, transport(fluid.transport,x,th)
end

"""
    transport(fluid, T, μ...)
    transport(fluid, T, μ::SVector)

Shorthand for `transport(fluid, GrandCanonicalPoint(T, μ...))`, returning the
thermodynamic and transport states. Plain numbers are grand-canonical
coordinates in the EOS unit convention.
"""
@inline transport(fluid::FluidModel, T::Real, μ::Real...) =
    transport(fluid, GrandCanonicalPoint(T, μ...))

@inline transport(fluid::FluidModel, T::Real, μ::SVector) =
    transport(fluid, GrandCanonicalPoint(T, μ))
