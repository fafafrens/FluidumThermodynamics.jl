
# Copy a small plain matrix into a static one so evaluation avoids heap arrays.
# Static matrices pass through; large matrices stay as given to bound compile time.
_static_matrix(A::StaticMatrix) = A
_static_matrix(A::AbstractMatrix) = length(A) <= 64 ? SMatrix{size(A,1),size(A,2)}(A) : A
_static_matrix(A) = A

"""
    AbstractEquationOfState{U<:AbstractUnitSystem}

EOS interface with a fixed unit convention and a required `unit_scale` field.
Custom models select `U` in their supertype, initialize the field with
`unit_scale(U())`, and construct specialized states with
`ThermodynamicState(eos,p,g,H)` to inherit that convention.
"""
abstract type AbstractEquationOfState{U<:AbstractUnitSystem} end

@inline unit_system(::AbstractEquationOfState{U}) where {U} = U()
@inline unit_scale(eos::AbstractEquationOfState) = eos.unit_scale

"""
    pressure(eos, x::GrandCanonicalPoint)

Fundamental EOS interface. A concrete EOS defines its pressure as a function of
grand-canonical coordinates. Optimized or tabulated EOS implementations may
also specialize `thermodynamic` directly.
"""
function pressure end

# EOS algebra is defined at the model level. ThermodynamicState itself is just
# evaluated data and deliberately has no symbolic/calculus algebra.
struct SumEOS{U,A<:AbstractEquationOfState{U},B<:AbstractEquationOfState{U},S} <: AbstractEquationOfState{U}
    a::A
    b::B
    unit_scale::S
    function SumEOS(a::AbstractEquationOfState{U}, b::AbstractEquationOfState{V}) where {U,V}
        U === V || throw(ArgumentError(
            "cannot combine EOS models with different unit systems ($U and $V); use convert_units first"))
        scale = unit_scale(a)
        new{U,typeof(a),typeof(b),typeof(scale)}(a,b,scale)
    end
end

struct ScaledEOS{U,T,E<:AbstractEquationOfState{U},S} <: AbstractEquationOfState{U}
    factor::T
    eos::E
    unit_scale::S
    function ScaledEOS(factor, eos::AbstractEquationOfState{U}) where {U}
        scale = unit_scale(eos)
        new{U,typeof(factor),typeof(eos),typeof(scale)}(factor,eos,scale)
    end
end

struct OppositeEOS{U,E<:AbstractEquationOfState{U},S} <: AbstractEquationOfState{U}
    eos::E
    unit_scale::S
    function OppositeEOS(eos::AbstractEquationOfState{U}) where {U}
        scale = unit_scale(eos)
        new{U,typeof(eos),typeof(scale)}(eos,scale)
    end
end

Base.:+(a::AbstractEquationOfState, b::AbstractEquationOfState) = SumEOS(a,b)
Base.:-(a::AbstractEquationOfState) = OppositeEOS(a)
Base.:-(a::AbstractEquationOfState, b::AbstractEquationOfState) = a + (-b)
Base.:*(a::Number, eos::AbstractEquationOfState) = ScaledEOS(a,eos)
Base.:*(eos::AbstractEquationOfState, a::Number) = a*eos
Base.:/(eos::AbstractEquationOfState, a::Number) = inv(a)*eos

@inline pressure(eos::SumEOS, x::GrandCanonicalPoint) = pressure(eos.a,x) + pressure(eos.b,x)
@inline pressure(eos::ScaledEOS, x::GrandCanonicalPoint) = eos.factor*pressure(eos.eos,x)
@inline pressure(eos::OppositeEOS, x::GrandCanonicalPoint) = -pressure(eos.eos,x)

# Evaluation in the original model's convention preserves optimized and
# tabulated EOS implementations. The target convention is encoded in the type.
struct ConvertedEOS{U<:AbstractUnitSystem,E<:AbstractEquationOfState,S} <: AbstractEquationOfState{U}
    eos::E
    unit_scale::S
    function ConvertedEOS(target::U, eos::E) where {U<:AbstractUnitSystem,E<:AbstractEquationOfState}
        scale = unit_scale(target)
        new{U,E,typeof(scale)}(eos,scale)
    end
end

function convert_units(target::AbstractUnitSystem, eos::AbstractEquationOfState)
    typeof(target) === typeof(unit_system(eos)) && return eos
    ConvertedEOS(target,eos)
end

function convert_units(target::AbstractUnitSystem, eos::ConvertedEOS)
    typeof(target) === typeof(unit_system(eos)) && return eos
    convert_units(target,eos.eos)
end

@inline function pressure(eos::ConvertedEOS, x::GrandCanonicalPoint)
    source, target = unit_system(eos.eos), unit_system(eos)
    original_point = convert_units(source,target,x)
    _conversion_factors(source,target).pressure * pressure(eos.eos,original_point)
end
