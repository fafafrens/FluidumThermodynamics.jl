
"""
    PolynomialMultiChargeEOS(a, chi; units=NaturalGeV())

Toy multi-charge EOS

    p(T,μ) = a T⁴ + 1/2 T² μᵀ chi μ.

It has nontrivial temperature-charge and charge-charge Hessian blocks and is
therefore useful for exercising multi-charge thermodynamics and inversion.
The supplied coefficients are numerical values in the selected convention,
with units pressure/energy⁴. Use `convert_units` to change the convention of
an existing model while preserving its physical predictions.

A plain matrix `chi` is copied into a static matrix, so evaluation does not
allocate; later changes to the original matrix do not affect the model.
"""
struct PolynomialMultiChargeEOS{U,A,K,S} <: AbstractEquationOfState{U}
    a::A
    chi::K
    unit_scale::S
end

function PolynomialMultiChargeEOS(a, chi; units::U=NaturalGeV()) where {U<:AbstractUnitSystem}
    scale = unit_scale(units)
    chi = _static_matrix(chi)
    PolynomialMultiChargeEOS{U,typeof(a),typeof(chi),typeof(scale)}(a,chi,scale)
end

@inline function pressure(eos::PolynomialMultiChargeEOS, x::GrandCanonicalPoint)
    size(eos.chi) == (length(x.μ),length(x.μ)) || throw(DimensionMismatch(
        "EOS and point have different numbers of charges"
    ))
    eos.a*x.T^4 + (x.T^2/2)*dot(x.μ, eos.chi*x.μ)
end

"""
    IdealQCD(Nc=3, Nf=2; units=NaturalGeV())

Ideal QCD EOS ported from Fluidum.jl, using natural GeV units by default.
`T` and the single chemical potential `μ` are in the
selected energy unit; pressure and energy density are in energy/length³.
The conversion uses
`ℏc = 0.1973261 GeV fm` when a convention with fm lengths is selected.

Supports zero or one chemical potential. The one-charge convention is exactly
the original Fluidum formula; it is not a generic choice of charge basis.
"""
struct IdealQCD{U,T,S} <: AbstractEquationOfState{U}
    a1::T
    a2::T
    a3::T
    unit_scale::S
end

function IdealQCD(Nc::Integer=3, Nf::Integer=2; units::AbstractUnitSystem=NaturalGeV())
    Nc > 0 && Nf >= 0 || throw(ArgumentError("Nc must be positive and Nf nonnegative"))
    a1 = 8*pi^2/15 * (Nc^2 - 1 + 7/4*Nc*Nf)
    a2 = 2.0*Nc*Nf/27
    a3 = 2*Nc*Nf/(81*pi^2)
    IdealQCD(a1,a2,a3; units)
end

function IdealQCD(a1,a2,a3; units::U=NaturalGeV()) where {U<:AbstractUnitSystem}
    a,b,c = promote(a1,a2,a3)
    scale = unit_scale(units)
    IdealQCD{U,typeof(a),typeof(scale)}(a,b,c,scale)
end

@inline function pressure(eos::IdealQCD, x::GrandCanonicalPoint)
    N = length(x.μ)
    N <= 1 || throw(DimensionMismatch("IdealQCD supports zero or one chemical potential"))
    μ = N == 0 ? zero(x.T) : x.μ[1]
    unit_scale(eos) * (eos.a1*x.T^4/24 + eos.a2*x.T^2*μ^2/4 + eos.a3*μ^4/24)
end

@inline function thermodynamic(eos::IdealQCD, x::GrandCanonicalPoint{T,0}) where {T}
    scale = unit_scale(eos)
    s = scale*eos.a1*x.T^3/6
    h = scale*eos.a1*x.T^2/2
    ThermodynamicState(eos,pressure(eos,x), SVector(s), SMatrix{1,1}(h))
end

@inline function thermodynamic(eos::IdealQCD, x::GrandCanonicalPoint{T,1}) where {T}
    t, μ = x.T, x.μ[1]
    scale = unit_scale(eos)
    s = scale*(eos.a1*t^3/6 + eos.a2*t*μ^2/2)
    n = scale*(eos.a2*t^2*μ/2 + eos.a3*μ^3/6)
    hTT = scale*(eos.a1*t^2/2 + eos.a2*μ^2/2)
    hTμ = scale*eos.a2*t*μ
    hμμ = scale*(eos.a2*t^2/2 + eos.a3*μ^2/2)
    ThermodynamicState(eos,pressure(eos,x), SVector(s,n), SMatrix{2,2}(hTT,hTμ,hTμ,hμμ))
end

"""
    grandcanonical(eos::IdealQCD, target::CanonicalPoint)

Analytic canonical inverse for zero or one conserved charge. For one charge,
`n = b*μ + c*μ³`, with `b = unit_scale(eos)*a2*T²/2` and
`c = unit_scale(eos)*a3/6`. Uses the real hyperbolic solution of the cubic,
avoiding subtraction of nearly equal cube roots at small density.
Requires nonnegative `b` and `c`. If both vanish, only zero density is
possible, and the representative `μ = 0` is selected.
"""
function grandcanonical(eos::IdealQCD, target::CanonicalPoint)
    N = length(target.n)
    N <= 1 || throw(DimensionMismatch("IdealQCD supports zero or one charge"))
    N == 0 && return GrandCanonicalPoint(target.T)
    n = target.n[1]
    b = unit_scale(eos)*eos.a2*target.T^2/2
    c = unit_scale(eos)*eos.a3/6
    μ = if b > 0 && c > 0
        r = sqrt(b/(3*c))
        2*r*sinh(asinh(3*n/(2*b*r))/3)
    elseif c == 0 && b > 0
        n/b
    elseif b == 0 && c > 0
        cbrt(n/c)
    elseif b == 0 && c == 0 && iszero(n)
        zero(float(n))
    else
        throw(DomainError((b,c,n),"no unique monotone IdealQCD canonical inverse for these coefficients and density"))
    end
    GrandCanonicalPoint(target.T,μ)
end

"""
    grandcanonical(eos::IdealQCD, target::MicrocanonicalPoint; guess=nothing, kwargs...)

At zero charge density, use `ε = unit_scale(eos)*a1*T⁴/8` to recover
temperature analytically, with `μ = 0` for a one-charge point.

For nonzero density, the coupled inverse is numerical. If no guess is supplied,
initialize temperature from the zero-density expression and chemical potential
from the analytic canonical inverse. Solver keywords apply to this numerical
branch and are forwarded to the existing microcanonical Newton solver.
"""
function grandcanonical(eos::IdealQCD, target::MicrocanonicalPoint;
                        guess=nothing, kwargs...)
    N = length(target.n)
    N <= 1 || throw(DimensionMismatch("IdealQCD supports zero or one charge"))
    t = sqrt(sqrt(8*target.energy_density/(unit_scale(eos)*eos.a1)))
    N == 0 && return GrandCanonicalPoint(t)
    iszero(target.n[1]) && return GrandCanonicalPoint(t,zero(t))
    initial = isnothing(guess) ? grandcanonical(eos,CanonicalPoint(t,target.n)) : guess
    grandcanonical(eos,target,initial; kwargs...)
end

"""
    thermodynamic(eos::IdealQCD, x::CanonicalPoint)
    thermodynamic(eos::IdealQCD, x::MicrocanonicalPoint; kwargs...)

Evaluate an IdealQCD state from canonical or microcanonical coordinates.
Canonical inversion is analytic. Microcanonical inversion is analytic at zero
charge density and numerical otherwise; see `grandcanonical`.

The returned state retains pressure derivatives with respect to `(T,μ...)`,
and inherits the EOS unit convention. It does not contain derivatives of a
canonical free energy or microcanonical entropy.
"""
thermodynamic(eos::IdealQCD, x::CanonicalPoint) =
    thermodynamic(eos,grandcanonical(eos,x))

thermodynamic(eos::IdealQCD, x::MicrocanonicalPoint; kwargs...) =
    thermodynamic(eos,grandcanonical(eos,x; kwargs...))

"""
    FluiduMEoS(; units=NaturalGeV())
    FluidumEOS(; units=NaturalGeV())

Temperature-only EOS fit ported from Fluidum.jl. Preserves the original
coefficients, temperature scales (154 and 100 MeV), and normalization
using `ℏc = 0.1973261 GeV fm`. Temperature and all evaluated quantities
follow the selected unit convention.

This is a zero-charge model: use `GrandCanonicalPoint(T)`. It does not
specify a finite-chemical-potential extension. Pressure and its first two
temperature derivatives are evaluated analytically.
"""
struct FluiduMEoS{U,T,S} <: AbstractEquationOfState{U}
    a1::T
    a2::T
    a3::T
    a4::T
    b1::T
    b2::T
    b3::T
    b4::T
    c::T
    d::T
    unit_scale::S
end

function FluiduMEoS(; units::AbstractUnitSystem=NaturalGeV())
    FluiduMEoS(
        -15.526548963383643, 18.6159584620131,
        -10.731808109698516, 2.7413302179949346,
        -3.3147904483107595, 5.310983721567554,
        -4.653922019495976, 1.8600649533271152,
        -1.0465330501411811, 0.09551531822245873;
        units,
    )
end

function FluiduMEoS(a1,a2,a3,a4,b1,b2,b3,b4,c,d;
                   units::U=NaturalGeV()) where {U<:AbstractUnitSystem}
    coefficients = promote(a1,a2,a3,a4,b1,b2,b3,b4,c,d)
    scale = unit_scale(units)
    FluiduMEoS{U,typeof(coefficients[1]),typeof(scale)}(coefficients...,scale)
end

"""Alias for `FluiduMEoS`."""
const FluidumEOS = FluiduMEoS

@inline function _fluidum_fit(eos::FluiduMEoS, T)
    invtemperature_MeV = inv(1000*energy_scale(unit_system(eos))*T)
    z, w = 154*invtemperature_MeV, 100*invtemperature_MeV
    numerator = evalpoly(z,(5.208957878352717,eos.a1,eos.a2,eos.a3,eos.a4))
    denominator = evalpoly(z,(1,eos.b1,eos.b2,eos.b3,eos.b4))
    exponential = exp(-muladd(eos.d^2,w,eos.c^2)*w)
    z,w,numerator,denominator,exponential
end

@inline function pressure(eos::FluiduMEoS, x::GrandCanonicalPoint)
    isempty(x.μ) || throw(DimensionMismatch("FluiduMEoS is a zero-charge EOS; use GrandCanonicalPoint(T)"))
    _,_,n,d,e = _fluidum_fit(eos,x.T)
    unit_scale(eos)*x.T^4*e*n/d
end

@inline function thermodynamic(eos::FluiduMEoS, x::GrandCanonicalPoint{T,0}) where {T}
    t = x.T
    z,w,n,d,e = _fluidum_fit(eos,t)
    n1 = evalpoly(z,(eos.a1,2*eos.a2,3*eos.a3,4*eos.a4))
    n2 = evalpoly(z,(2*eos.a2,6*eos.a3,12*eos.a4))
    d1 = evalpoly(z,(eos.b1,2*eos.b2,3*eos.b3,4*eos.b4))
    d2 = evalpoly(z,(2*eos.b2,6*eos.b3,12*eos.b4))
    # Derivatives of the rational fit with respect to z = 154 MeV / T.
    invd = inv(d)
    r = n*invd
    r1 = muladd(-r,d1,n1)*invd
    r2 = muladd(-2*r1,d1,muladd(-r,d2,n2))*invd
    # Temperature derivatives scaled by T and T², respectively.
    loge1 = w*muladd(2*eos.d^2,w,eos.c^2)
    loge2 = -w*muladd(6*eos.d^2,w,2*eos.c^2)
    # Combine product-rule terms before applying the shared exp/T² prefactor.
    # This avoids reconstructing f, T*f′, and T²*f″ separately.
    t2 = t*t
    prefactor = unit_scale(eos)*e*t2
    zr1 = z*r1
    p = prefactor*t2*r
    s = prefactor*t*muladd(4+loge1,r,-zr1)
    h = prefactor*muladd(muladd(loge1,loge1+8,loge2+12),r,
                        muladd(z*z,r2,-(6+2*loge1)*zr1))
    ThermodynamicState(eos,p,SVector(s),SMatrix{1,1}(h))
end
