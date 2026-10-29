"""
    ThermodynamicPerturbationState(eos, p, gradient, hessian, third)
    ThermodynamicPerturbationState(p, gradient, hessian, third; units=NaturalGeV())

Thermodynamic information for linearised (perturbative) hydrodynamics: a
[`ThermodynamicState`](@ref) extended by the fully symmetric third-derivative
tensor of the pressure in grand-canonical coordinates `y = (T, μ₁, ..., μₙ)`,

    third[i,j,k] = ∂³p/∂yᵢ∂yⱼ∂yₖ .

The perturbation equations around a hydrodynamic background contain the
derivative of the Hessian with respect to the background fields, which is
where this tensor enters (for a charge-free fluid, `third[1,1,1] = p‴(T)`).
Obtained from [`thermodynamic_perturbation`](@ref). Every accessor of
`ThermodynamicState` applies; `ThermodynamicState(th)` drops the tensor.
"""
struct ThermodynamicPerturbationState{U<:AbstractUnitSystem,T,D,G,H,C} <: AbstractThermodynamicState{U,T,D}
    pressure::T
    gradient::G
    hessian::H
    third::C
end

function ThermodynamicPerturbationState(p, g::SVector{D,G}, H::SMatrix{D,D,V},
                                        C::SArray{Tuple{D,D,D},W};
                                        units::U=NaturalGeV()) where {D,G,V,W,U<:AbstractUnitSystem}
    T = promote_type(typeof(p), G, V, W)
    gradient = SVector{D,T}(g)
    hessian = SMatrix{D,D,T}(H)
    third = SArray{Tuple{D,D,D},T}(C)
    ThermodynamicPerturbationState{U,T,D,typeof(gradient),typeof(hessian),typeof(third)}(
        T(p),gradient,hessian,third)
end

ThermodynamicPerturbationState(eos::AbstractEquationOfState, p, g, H, C) =
    ThermodynamicPerturbationState(p,g,H,C; units=unit_system(eos))

"""
    ThermodynamicState(th::ThermodynamicPerturbationState)

The second-order state with the same pressure, gradient and Hessian.
"""
ThermodynamicState(th::ThermodynamicPerturbationState{U}) where {U} =
    ThermodynamicState(th.pressure,th.gradient,th.hessian; units=U())

"""
    third_derivatives(th::ThermodynamicPerturbationState)
    third_derivative(th, i, j, k)

The tensor `∂³p/∂yᵢ∂yⱼ∂yₖ` in grand-canonical coordinates, or one of its entries
(`1` is the temperature, `i+1` the `i`-th chemical potential).
"""
@inline third_derivatives(th::ThermodynamicPerturbationState) = th.third
@inline third_derivative(th::ThermodynamicPerturbationState, i::Integer, j::Integer, k::Integer) = th.third[i,j,k]

function convert_units(target::AbstractUnitSystem, th::ThermodynamicPerturbationState)
    typeof(target) === typeof(unit_system(th)) && return th
    f = _conversion_factors(unit_system(th),target)
    ThermodynamicPerturbationState(f.pressure*th.pressure, f.density*th.gradient,
                                   f.hessian*th.hessian, f.third*th.third; units=target)
end

"""
    thermodynamic_perturbation(eos, x::GrandCanonicalPoint)
    thermodynamic_perturbation(eos, T, μ...)

Evaluate the EOS at `x` to third order: a [`ThermodynamicPerturbationState`](@ref)
with the pressure, its gradient `(s, n)`, its Hessian and its third-derivative
tensor. The generic method differentiates `pressure(eos, x)` with nested
ForwardDiff; `IdealQCD` is analytic, the EOS algebra (`+`, `*`, `-`,
`convert_units`) and `HeavyQuark` compose their parts, and `TabulatedEOS`
throws, since a cubic interpolant has no meaningful third derivative.
"""
function thermodynamic_perturbation(eos::AbstractEquationOfState, x::GrandCanonicalPoint)
    y = coordinates(x)
    D = length(y)
    f(z) = pressure(eos, grandcanonical_from_coordinates(z))

    p = f(y)
    g = SVector{D}(ForwardDiff.gradient(f,y))
    H = SMatrix{D,D}(ForwardDiff.hessian(f,y))
    # Jacobian of the vectorised Hessian: column-major, so entry (i + D(j-1), k) is ∂ᵢ∂ⱼ∂ₖp.
    hvec(z) = vec(ForwardDiff.hessian(f,z))
    J = ForwardDiff.jacobian(hvec,y)
    C = SArray{Tuple{D,D,D},eltype(J)}(Tuple(J))

    ThermodynamicPerturbationState(eos,p,g,H,C)
end

@inline thermodynamic_perturbation(eos::AbstractEquationOfState, T::Real, μ::Real...) =
    thermodynamic_perturbation(eos, GrandCanonicalPoint(T, μ...))

@inline thermodynamic_perturbation(eos::AbstractEquationOfState, T::Real, μ::SVector) =
    thermodynamic_perturbation(eos, GrandCanonicalPoint(T, μ))

@inline function thermodynamic_perturbation(eos::SumEOS, x::GrandCanonicalPoint)
    a, b = thermodynamic_perturbation(eos.a,x), thermodynamic_perturbation(eos.b,x)
    ThermodynamicPerturbationState(eos,a.pressure+b.pressure, a.gradient+b.gradient,
                                   a.hessian+b.hessian, a.third+b.third)
end

@inline function thermodynamic_perturbation(eos::ScaledEOS, x::GrandCanonicalPoint)
    th = thermodynamic_perturbation(eos.eos,x)
    ThermodynamicPerturbationState(eos,eos.factor*th.pressure, eos.factor*th.gradient,
                                   eos.factor*th.hessian, eos.factor*th.third)
end

@inline function thermodynamic_perturbation(eos::OppositeEOS, x::GrandCanonicalPoint)
    th = thermodynamic_perturbation(eos.eos,x)
    ThermodynamicPerturbationState(eos,-th.pressure, -th.gradient, -th.hessian, -th.third)
end

@inline function thermodynamic_perturbation(eos::ConvertedEOS, x::GrandCanonicalPoint)
    original_point = convert_units(unit_system(eos.eos),unit_system(eos),x)
    th = thermodynamic_perturbation(eos.eos,original_point)
    convert_units(unit_system(eos),th)
end
