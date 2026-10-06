"""
    AbstractSecondOrderTransport

Interface for additional second-order constitutive coefficients. Implement
`second_order_state(model, x, th, leading)`, returning a `SecondOrderState`.
`leading` contains the already evaluated shear, bulk, and charge sectors,
including their relaxation times; `leading.second_order === nothing`.
"""
abstract type AbstractSecondOrderTransport end

"""
    NoSecondOrderTransport()

Supply no additional second-order coefficients. Existing relaxation times
remain part of the shear, bulk, and charge states.
"""
struct NoSecondOrderTransport <: AbstractSecondOrderTransport end

"""
    SecondOrderState(basis::Symbol; shear=(;), bulk=(;), charge=(;))

Evaluated second-order coefficients grouped by their destination equation.
Each sector is a `NamedTuple`; names and the hydrodynamic frame, signs,
normalization, and units are defined by the explicitly named `basis`.
The package does not interpret the names or infer coefficients from an EOS.

Shear and bulk entries can be scalars or numeric arrays. Charge entries must
be numeric arrays with at least one axis for the output charge. Every array
axis denotes a conserved-charge index, in the order of `x.μ`, and is checked
against the number of charges by `transport`. Arrays contain coefficients,
not spacetime tensors. The basis defines the meaning and order of the axes.

For example, a shear/bulk coupling can be stored as
`shear=(lambda_piPi=...,)`, diffusion-expansion couplings as
`charge=(delta_VV=matrix,)`, and nonlinear diffusion couplings as rank-three
arrays. Omitted entries are unspecified, not implicit zeros. No universal
coefficient list, symmetry, or physical closure is imposed.

Array storage is retained by reference.
Use immutable static arrays or copy mutable arrays when independent ownership
is needed.
"""
struct SecondOrderState{S<:NamedTuple,B<:NamedTuple,C<:NamedTuple}
    basis::Symbol
    shear::S
    bulk::B
    charge::C

    function SecondOrderState(basis::Symbol, shear::S, bulk::B, charge::C) where {
            S<:NamedTuple,B<:NamedTuple,C<:NamedTuple}
        _check_sector(_check_coefficient, :shear, shear)
        _check_sector(_check_coefficient, :bulk, bulk)
        _check_sector(_check_coefficient, :charge, charge)
        new{S,B,C}(basis,shear,bulk,charge)
    end
end

# Apply `check(sector, name, value)` to every entry of a sector. `map` over the
# names and values is unrolled with each entry's concrete type; iterating
# `pairs` of a NamedTuple with mixed value types is not type-stable.
@inline function _check_sector(check::F, sector::Symbol, coefficients::NamedTuple) where {F}
    map((name, value) -> check(sector, name, value), keys(coefficients), values(coefficients))
    nothing
end

@inline function _check_coefficient(sector, name, value::AbstractArray{<:Number})
    ndims(value) > 0 || throw(ArgumentError("$sector.$name must have at least one charge axis"))
    nothing
end

@inline function _check_coefficient(sector, name, value::Number)
    sector === :charge && throw(ArgumentError("$sector.$name must be a numeric array over charge indices"))
    nothing
end

function _check_coefficient(sector, name, value)
    throw(ArgumentError(sector === :charge ?
        "$sector.$name must be a numeric array over charge indices" :
        "$sector.$name must be a number or numeric array over charge indices"))
end

SecondOrderState(basis::Symbol; shear::NamedTuple=(;), bulk::NamedTuple=(;),
                 charge::NamedTuple=(;)) = SecondOrderState(basis,shear,bulk,charge)

"""
    ConstantSecondOrderTransport(state::SecondOrderState)
    ConstantSecondOrderTransport(basis::Symbol; shear=(;), bulk=(;), charge=(;))

A model with prescribed, state-independent second-order coefficients.
"""
struct ConstantSecondOrderTransport{S<:SecondOrderState} <: AbstractSecondOrderTransport
    state::S
end

ConstantSecondOrderTransport(basis::Symbol; kwargs...) =
    ConstantSecondOrderTransport(SecondOrderState(basis; kwargs...))

"""
    FunctionalSecondOrderTransport(f)

A state-dependent model evaluated as `f(x, th, leading)`. Return a
`SecondOrderState`. The callback can use temperature, chemical potentials,
EOS derivatives, viscosities, conductivities, and relaxation times without
re-evaluating the EOS or leading transport sectors.
"""
struct FunctionalSecondOrderTransport{F} <: AbstractSecondOrderTransport
    f::F
end

"""
    second_order_state(model, x, th, leading)

Evaluate additional second-order transport coefficients. Custom subtypes of
`AbstractSecondOrderTransport` specialize this method. Evaluation through
`transport` also validates the returned state and its charge dimensions.
"""
function second_order_state end

@inline second_order_state(::NoSecondOrderTransport, x, th, leading) = nothing
@inline second_order_state(model::ConstantSecondOrderTransport, x, th, leading) = model.state
@inline second_order_state(model::FunctionalSecondOrderTransport, x, th, leading) =
    model.f(x,th,leading)

@inline _validate_second_order(::Nothing, x::GrandCanonicalPoint) = nothing

function _validate_second_order(state::SecondOrderState, x::GrandCanonicalPoint{T,N}) where {T,N}
    check(sector, name, value) = _check_charge_axes(sector, name, value, Val(N))
    _check_sector(check, :shear, state.shear)
    _check_sector(check, :bulk, state.bulk)
    _check_sector(check, :charge, state.charge)
    state
end

@inline _check_charge_axes(sector, name, value, ::Val) = nothing

@inline function _check_charge_axes(sector, name, value::AbstractArray, ::Val{N}) where {N}
    all(axis -> axis == Base.OneTo(N), axes(value)) || throw(DimensionMismatch(
        "$sector.$name must have $N entries along every charge axis; got axes $(axes(value))"))
    nothing
end

_validate_second_order(state, x::GrandCanonicalPoint) =
    throw(ArgumentError("second_order_state must return SecondOrderState or nothing"))
