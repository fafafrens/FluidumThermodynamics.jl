
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
Coupled charge-transport model. Conductivity and relaxation time are matrices,
so off-diagonal transport among conserved charges is supported directly.
"""
struct ConstantChargeTransport{K,R} <: AbstractChargeTransport
    kappa::K
    tau::R
end

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
    Z = SMatrix{N,N}(ntuple(_ -> z, N*N))
    ChargeState(Z,Z)
end

@inline function charge_state(model::ConstantChargeTransport, x::GrandCanonicalPoint{T,N}, th) where {T,N}
    size(model.kappa) == (N,N) && size(model.tau) == (N,N) ||
        throw(DimensionMismatch("charge transport matrices must match the number of charges"))
    ChargeState(model.kappa,model.tau)
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
