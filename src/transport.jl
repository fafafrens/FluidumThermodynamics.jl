
abstract type AbstractShearTransport end
abstract type AbstractBulkTransport end
abstract type AbstractChargeTransport end

struct ZeroShearTransport <: AbstractShearTransport end
struct ZeroBulkTransport <: AbstractBulkTransport end
struct ZeroChargeTransport <: AbstractChargeTransport end

"""Simple η/s model with τπ = ℏc·η/[Cτ(ε+p)], in the length unit of the convention (`hbarc`)."""
struct ConstantEtaOverS{T} <: AbstractShearTransport
    eta_over_s::T
    Ctau::T
end

"""Simple ζ/s model with τΠ = ℏc·ζ/[Cτ(ε+p)], in the length unit of the convention (`hbarc`)."""
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

ShearState(eta, tau) = ShearState(promote(eta,tau)...)
BulkState(zeta, tau) = BulkState(promote(zeta,tau)...)

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
    # η/(Ctau·w) is an inverse energy; hbarc makes it a length of the convention.
    tau = eta/(model.Ctau*enthalpy(x,th))*hbarc(unit_system(th))
    ShearState(eta,tau)
end

@inline function bulk_state(::ZeroBulkTransport, x, th)
    z = zero(th.pressure)
    BulkState(z,z)
end

@inline function bulk_state(model::ConstantZetaOverS, x, th)
    zeta = model.zeta_over_s*entropy_density(th)
    tau = zeta/(model.Ctau*enthalpy(x,th))*hbarc(unit_system(th))
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

# Quantities computed with ℏ = c = 1 carry lengthᵇ through hbarc(units)^b (see `hbarc`).
# κ = k̂·T²: energy² to length⁻².
@inline _kappa_from_T2(k̂, T, units::AbstractUnitSystem) = k̂*T^2/hbarc(units)^2

# κ = DₛT·n/T: length⁻³·energy⁻¹ to length⁻².
@inline _kappa_from_DsT(DsT, n, T, units::AbstractUnitSystem) = DsT*n/T*hbarc(units)

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
