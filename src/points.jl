
"""
    GrandCanonicalPoint(T, μ)

Grand-canonical coordinates

    (T, μ₁, ..., μₙ).

These coordinates select a point at which the pressure is evaluated. They are
not themselves the complete thermodynamic state.
"""
struct GrandCanonicalPoint{T,N}
    T::T
    μ::SVector{N,T}
end

GrandCanonicalPoint(T::R) where {R} =
    GrandCanonicalPoint{R,0}(T, SVector{0,R}())

function GrandCanonicalPoint(T::S, μ::SVector{N,U}) where {S,U,N}
    R = promote_type(S,U)
    GrandCanonicalPoint{R,N}(R(T), SVector{N,R}(μ))
end

GrandCanonicalPoint(T, μ...) = GrandCanonicalPoint(T, SVector(μ...))

"""
    CanonicalPoint(T, n)

Canonical local thermodynamic coordinates `(T, n₁, ..., nₙ)`.
"""
struct CanonicalPoint{T,N}
    T::T
    n::SVector{N,T}
end

CanonicalPoint(T::R) where {R} = CanonicalPoint{R,0}(T, SVector{0,R}())
CanonicalPoint(T, n...) = CanonicalPoint(T, SVector(n...))

function CanonicalPoint(T::S, n::SVector{N,U}) where {S,U,N}
    R = promote_type(S,U)
    CanonicalPoint{R,N}(R(T), SVector{N,R}(n))
end

"""
    MicrocanonicalPoint(energy_density, n)

Microcanonical local thermodynamic coordinates `(ε, n₁, ..., nₙ)`.

Using energy density here keeps the ensemble terminology literal: the
microcanonical representation fixes energy and conserved charges. Entropy is a
derived thermodynamic quantity.
"""
struct MicrocanonicalPoint{T,N}
    energy_density::T
    n::SVector{N,T}
end

MicrocanonicalPoint(ε::R) where {R} = MicrocanonicalPoint{R,0}(ε, SVector{0,R}())
MicrocanonicalPoint(ε, n...) = MicrocanonicalPoint(ε, SVector(n...))

function MicrocanonicalPoint(ε::S, n::SVector{N,U}) where {S,U,N}
    R = promote_type(S,U)
    MicrocanonicalPoint{R,N}(R(ε), SVector{N,R}(n))
end

@inline coordinates(x::GrandCanonicalPoint) = SVector(x.T, x.μ...)

@inline function grandcanonical_from_coordinates(y::SVector{D,T}) where {D,T}
    if D == 1
        return GrandCanonicalPoint(y[1])
    end
    μ = SVector{D-1,T}(ntuple(i -> y[i+1], D-1))
    GrandCanonicalPoint(y[1], μ)
end

function convert_units(target::AbstractUnitSystem, source::AbstractUnitSystem,
                       x::GrandCanonicalPoint)
    factor = _conversion_factors(source,target).energy
    GrandCanonicalPoint(factor*x.T, factor*x.μ)
end

function convert_units(target::AbstractUnitSystem, source::AbstractUnitSystem,
                       x::CanonicalPoint)
    f = _conversion_factors(source,target)
    CanonicalPoint(f.energy*x.T, f.density*x.n)
end

function convert_units(target::AbstractUnitSystem, source::AbstractUnitSystem,
                       x::MicrocanonicalPoint)
    f = _conversion_factors(source,target)
    MicrocanonicalPoint(f.pressure*x.energy_density, f.density*x.n)
end
