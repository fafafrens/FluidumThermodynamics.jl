
"""
    ThermodynamicState(eos, p, gradient, hessian)
    ThermodynamicState(p, gradient, hessian; units=NaturalGeV())

Thermodynamic information obtained by evaluating an EOS at a chosen point.
In grand-canonical coordinates

    x = (T, μ₁, ..., μₙ),

it stores

    p(x),  ∇p(x),  ∇²p(x).

The gradient is the Legendre-dual vector

    ∇p = (s, n₁, ..., nₙ),

not another grand-canonical point. The lower-right Hessian block is the charge
susceptibility matrix `χᵢⱼ = ∂nᵢ/∂μⱼ`.
The unit convention is encoded in the state type. EOS evaluation inherits it
from the model; the three-argument constructor is for manually supplied data.
"""
struct ThermodynamicState{U<:AbstractUnitSystem,T,D,G,H}
    pressure::T
    gradient::G
    hessian::H
end

function ThermodynamicState(p, g::SVector{D,G}, H::SMatrix{D,D,V};
                            units::U=NaturalGeV()) where {D,G,V,U<:AbstractUnitSystem}
    T = promote_type(typeof(p), G, V)
    gradient = SVector{D,T}(g)
    hessian = SMatrix{D,D,T}(H)
    ThermodynamicState{U,T,D,typeof(gradient),typeof(hessian)}(T(p),gradient,hessian)
end

ThermodynamicState(eos::AbstractEquationOfState, p, g, H) =
    ThermodynamicState(p,g,H; units=unit_system(eos))

@inline unit_system(::ThermodynamicState{U}) where {U} = U()

function convert_units(target::AbstractUnitSystem, th::ThermodynamicState)
    typeof(target) === typeof(unit_system(th)) && return th
    f = _conversion_factors(unit_system(th),target)
    ThermodynamicState(f.pressure*th.pressure, f.density*th.gradient,
                       f.hessian*th.hessian; units=target)
end

"""
    thermodynamic(eos, x)

Generic second-order ForwardDiff fallback. A concrete/tabulated EOS can
specialize this method while returning the same `ThermodynamicState` interface.
"""
function thermodynamic(eos::AbstractEquationOfState, x::GrandCanonicalPoint)
    y = coordinates(x)
    D = length(y)
    f(z) = pressure(eos, grandcanonical_from_coordinates(z))

    p = f(y)
    g = SVector{D}(ForwardDiff.gradient(f,y))
    H = SMatrix{D,D}(ForwardDiff.hessian(f,y))

    ThermodynamicState(eos,p,g,H)
end

"""
    thermodynamic(eos, T, μ...)
    thermodynamic(eos, T, μ::SVector)

Shorthand for `thermodynamic(eos, GrandCanonicalPoint(T, μ...))`. Plain numbers
are always grand-canonical coordinates in the EOS unit convention; omitted
chemical potentials give a zero-charge point.
"""
@inline thermodynamic(eos::AbstractEquationOfState, T::Real, μ::Real...) =
    thermodynamic(eos, GrandCanonicalPoint(T, μ...))

@inline thermodynamic(eos::AbstractEquationOfState, T::Real, μ::SVector) =
    thermodynamic(eos, GrandCanonicalPoint(T, μ))

@inline function thermodynamic(eos::SumEOS, x::GrandCanonicalPoint)
    a, b = thermodynamic(eos.a,x), thermodynamic(eos.b,x)
    ThermodynamicState(eos,a.pressure+b.pressure, a.gradient+b.gradient, a.hessian+b.hessian)
end

@inline function thermodynamic(eos::ScaledEOS, x::GrandCanonicalPoint)
    th = thermodynamic(eos.eos,x)
    ThermodynamicState(eos,eos.factor*th.pressure, eos.factor*th.gradient, eos.factor*th.hessian)
end

@inline function thermodynamic(eos::OppositeEOS, x::GrandCanonicalPoint)
    th = thermodynamic(eos.eos,x)
    ThermodynamicState(eos,-th.pressure, -th.gradient, -th.hessian)
end

@inline function thermodynamic(eos::ConvertedEOS, x::GrandCanonicalPoint)
    original_point = convert_units(unit_system(eos.eos),unit_system(eos),x)
    th = thermodynamic(eos.eos,original_point)
    convert_units(unit_system(eos),th)
end

@inline entropy_density(th::ThermodynamicState) = th.gradient[1]

@inline function charge_densities(th::ThermodynamicState{U,T,D}) where {U,T,D}
    N = D - 1
    SVector{N,T}(ntuple(i -> th.gradient[i+1], N))
end

@inline charge_density(th::ThermodynamicState, i::Integer) = th.gradient[i+1]
@inline susceptibility(th::ThermodynamicState, i::Integer, j::Integer) = th.hessian[i+1,j+1]

@inline function susceptibilities(th::ThermodynamicState{U,T,D}) where {U,T,D}
    N = D - 1
    SMatrix{N,N,T}(ntuple(k -> begin
        i = (k-1) % N + 1
        j = (k-1) ÷ N + 1
        th.hessian[i+1,j+1]
    end, N*N))
end

"""
    energy_density(x, th)

Legendre transform

    ε = -p + T s + μ⋅n.
"""
@inline function energy_density(x::GrandCanonicalPoint, th::ThermodynamicState)
    -th.pressure + x.T*entropy_density(th) + dot(x.μ, charge_densities(th))
end

"""
    enthalpy(x, th)

Enthalpy density `w = ε + p = T s + μ⋅n`, evaluated without the cancellation
in `-p + T s + μ⋅n`.
"""
@inline enthalpy(x::GrandCanonicalPoint, th::ThermodynamicState) =
    x.T*entropy_density(th) + dot(x.μ, charge_densities(th))

"""
    trace_anomaly(x, th)

Trace of the energy-momentum tensor, `ε - 3p`. It vanishes for a conformal EOS.
Divide by `T^4` for the dimensionless interaction measure in natural units.
"""
@inline trace_anomaly(x::GrandCanonicalPoint, th::ThermodynamicState) =
    energy_density(x,th) - 3*th.pressure

"""
    pressure_derivatives(x, th)

Derivatives of the pressure in microcanonical variables,

    (∂p/∂ε|ₙ, ∂p/∂n₁|_ε, …, ∂p/∂nₙ|_ε).

They follow from the stored gradient and Hessian: with `y = (T, μ)`,
`∂ε/∂y = yᵀ ∇²p` and `∂n/∂y` are the charge rows of `∇²p`. The Jacobian
`∂(ε,n)/∂(T,μ)` must be nonsingular.
"""
@inline function pressure_derivatives(x::GrandCanonicalPoint, th::ThermodynamicState{U,T,D}) where {U,T,D}
    y = SVector{D,T}(x.T, x.μ...)
    H = th.hessian
    dε = H'*y
    J = SMatrix{D,D,T}(ntuple(Val(D*D)) do index
        i = mod1(index,D)
        j = (index-1) ÷ D + 1
        i == 1 ? dε[j] : H[i,j]
    end)
    J' \ th.gradient
end

"""
    speed_of_sound_squared(x, th)

Adiabatic speed of sound squared at fixed entropy per charge,

    c_s² = ∂p/∂ε|ₙ + Σᵢ nᵢ/(ε+p) ∂p/∂nᵢ|_ε.

With no charges this is `s/(T ∂s/∂T)`. Velocities are in units of `c`.
"""
@inline function speed_of_sound_squared(x::GrandCanonicalPoint, th::ThermodynamicState)
    dp = pressure_derivatives(x,th)
    n = charge_densities(th)
    dp[1] + dot(n, dp[2:end])/enthalpy(x,th)
end

for f in (:enthalpy, :trace_anomaly, :pressure_derivatives, :speed_of_sound_squared)
    @eval @inline $f(eos::AbstractEquationOfState, x::GrandCanonicalPoint) = $f(x, thermodynamic(eos,x))
end

# Representation changes when the state has already been evaluated.
@inline canonical(x::GrandCanonicalPoint, th::ThermodynamicState) =
    CanonicalPoint(x.T, charge_densities(th))

@inline microcanonical(x::GrandCanonicalPoint, th::ThermodynamicState) =
    MicrocanonicalPoint(energy_density(x,th), charge_densities(th))

# Convenience forms that evaluate the EOS once.
@inline function canonical(eos::AbstractEquationOfState, x::GrandCanonicalPoint)
    th = thermodynamic(eos,x)
    canonical(x,th)
end

@inline function microcanonical(eos::AbstractEquationOfState, x::GrandCanonicalPoint)
    th = thermodynamic(eos,x)
    microcanonical(x,th)
end
