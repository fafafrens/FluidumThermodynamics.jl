"""
    TabulatedEOS(T, p, s, dsdT; units=NaturalGeV())
    TabulatedEOS(T, p, s; units=NaturalGeV())
    TabulatedEOS(eos, T)

Zero-charge EOS interpolated from a temperature table.

Nodes `T` must be positive and strictly increasing; they need not be uniform.
`p`, `s`, and `dsdT` are the pressure, entropy density, and `∂s/∂T` at each
node, in the convention selected by `units`. The table interpolates the single
function `f = p/T⁴` with quintic Hermite polynomials that match `f`, `f′`, and
`f″` at every node. Pressure, entropy, and `∂s/∂T` are all derivatives of this
one interpolant, so the table is thermodynamically consistent and `p` is twice
continuously differentiable. A conformal EOS (`p ∝ T⁴`) is reproduced exactly.

Without `dsdT`, it is estimated by fourth-order finite differences of `s/T³`
and at least five nodes are required. The table then matches `p` and `s` at the
nodes, but `∂s/∂T` is less accurate. Supply it when available; lattice tables
that list the speed of sound give `∂s/∂T = s/(T c_s²)`. `TabulatedEOS(eos, T)`
samples an existing zero-charge EOS, including its exact `∂s/∂T`, and inherits
its unit convention.

Stability requires `∂s/∂T > 0`, so energy density increases with temperature.
The table is not extrapolated: evaluation outside `[first(T), last(T)]` throws
a `DomainError`.
"""
struct TabulatedEOS{U,T,S} <: AbstractEquationOfState{U}
    temperature::Vector{T}
    # p/T⁴ and its first two temperature derivatives at the nodes.
    f::Vector{T}
    df::Vector{T}
    d2f::Vector{T}
    # Energy density at the nodes, used to bracket the microcanonical inverse.
    energy_density::Vector{T}
    unit_scale::S
end

function TabulatedEOS(temperature::AbstractVector, p::AbstractVector,
                      s::AbstractVector, dsdT::AbstractVector;
                      units::U=NaturalGeV()) where {U<:AbstractUnitSystem}
    n = length(temperature)
    n >= 2 || throw(ArgumentError("a table needs at least two temperatures"))
    length(p) == length(s) == length(dsdT) == n ||
        throw(DimensionMismatch("temperature, pressure, entropy, and dsdT tables must have equal length"))
    R = float(promote_type(eltype(temperature),eltype(p),eltype(s),eltype(dsdT)))
    t, p, s, c = R.(temperature), R.(p), R.(s), R.(dsdT)
    all(isfinite, t) && all(isfinite, p) && all(isfinite, s) && all(isfinite, c) ||
        throw(ArgumentError("table entries must be finite"))
    first(t) > 0 || throw(ArgumentError("temperatures must be positive"))
    all(i -> t[i] < t[i+1], 1:n-1) ||
        throw(ArgumentError("temperatures must be strictly increasing"))
    all(>(0), c) || throw(ArgumentError("dsdT must be positive for a stable EOS"))
    # f = p/T⁴; f′ and f″ follow from p′ = s and p″ = ∂s/∂T.
    f = @. p/t^4
    df = @. (s - 4*p/t)/t^4
    d2f = @. (c - 8*s/t + 20*p/t^2)/t^4
    ε = @. -p + t*s
    all(i -> ε[i] < ε[i+1], 1:n-1) ||
        throw(ArgumentError("energy density must increase with temperature"))
    scale = unit_scale(units)
    TabulatedEOS{U,R,typeof(scale)}(t,f,df,d2f,ε,scale)
end

function TabulatedEOS(temperature::AbstractVector, p::AbstractVector, s::AbstractVector;
                      units::AbstractUnitSystem=NaturalGeV())
    n = length(temperature)
    n >= 5 || throw(ArgumentError("estimating dsdT needs at least five temperatures"))
    length(s) == n || throw(DimensionMismatch("temperature and entropy tables must have equal length"))
    # Differentiate the smooth ratio s/T³ rather than s itself:
    # ∂s/∂T = T³ ∂(s/T³)/∂T + 3s/T.
    t = float.(temperature)
    g = _derivative(t, s ./ t.^3)
    TabulatedEOS(temperature,p,s,(@. t^3*g + 3*s/t); units)
end

function TabulatedEOS(eos::AbstractEquationOfState, temperature::AbstractVector)
    states = [thermodynamic(eos,GrandCanonicalPoint(t)) for t in temperature]
    TabulatedEOS(temperature,
                 [th.pressure for th in states],
                 [entropy_density(th) for th in states],
                 [th.hessian[1,1] for th in states];
                 units=unit_system(eos))
end

# Fourth-order finite differences on a nonuniform grid: the derivative at each
# node of the quartic through five neighbouring nodes, shifted at the ends.
function _derivative(x::AbstractVector, y::AbstractVector)
    n = length(x)
    d = similar(y, float(promote_type(eltype(x),eltype(y))))
    for i in 1:n
        nodes = clamp(i-2,1,n-4):clamp(i-2,1,n-4)+4
        t = x[i]
        total = zero(eltype(d))
        for a in nodes
            # Derivative of the Lagrange basis polynomial for node a at t.
            weight = zero(eltype(d))
            for b in nodes
                b == a && continue
                term = inv(x[a]-x[b])
                for c in nodes
                    (c == a || c == b) && continue
                    term *= (t-x[c])/(x[a]-x[c])
                end
                weight += term
            end
            total += weight*y[a]
        end
        d[i] = total
    end
    d
end

@inline function _table_cell(nodes::AbstractVector, value)
    i = searchsortedlast(nodes,value)
    clamp(i,1,length(nodes)-1)
end

# Quintic Hermite interpolant of f on cell i, with its first two derivatives.
@inline function _hermite(eos::TabulatedEOS, i::Integer, T)
    @inbounds begin
        a, b = eos.temperature[i], eos.temperature[i+1]
        h = b - a
        y0, y1 = eos.f[i], eos.f[i+1]
        d0, d1 = h*eos.df[i], h*eos.df[i+1]
        s0, s1 = h^2*eos.d2f[i], h^2*eos.d2f[i+1]
    end
    Δ = y1 - y0
    c3 = 10Δ - 6d0 - 4d1 - (3s0 - s1)/2
    c4 = -15Δ + 8d0 + 7d1 + (3s0 - 2s1)/2
    c5 = 6Δ - 3d0 - 3d1 - (s0 - s1)/2
    t = (T - a)/h
    f = evalpoly(t,(y0,d0,s0/2,c3,c4,c5))
    df = evalpoly(t,(d0,s0,3c3,4c4,5c5))/h
    d2f = evalpoly(t,(s0,6c3,12c4,20c5))/h^2
    f, df, d2f
end

@inline function _tabulated_state(eos::TabulatedEOS, T)
    lo, hi = first(eos.temperature), last(eos.temperature)
    lo <= T <= hi || throw(DomainError(T,"temperature is outside the table range [$lo, $hi]"))
    f, df, d2f = _hermite(eos,_table_cell(eos.temperature,T),T)
    T2 = T*T
    p = T2*T2*f
    s = T2*T*muladd(T,df,4f)
    dsdT = T2*muladd(T,muladd(T,d2f,8df),12f)
    p, s, dsdT
end

@inline function _check_zero_charge(x::GrandCanonicalPoint)
    isempty(x.μ) || throw(DimensionMismatch("TabulatedEOS is a zero-charge EOS; use GrandCanonicalPoint(T)"))
end

@inline function thermodynamic(eos::TabulatedEOS, x::GrandCanonicalPoint)
    _check_zero_charge(x)
    p, s, dsdT = _tabulated_state(eos,x.T)
    ThermodynamicState(eos,p,SVector(s),SMatrix{1,1}(dsdT))
end

# The cubic Hermite interpolant has a piecewise-linear, discontinuous third derivative:
# not a usable ∂³p/∂T³. Perturbation matrices need an analytic EOS.
function thermodynamic_perturbation(eos::TabulatedEOS, x::GrandCanonicalPoint)
    throw(ArgumentError("TabulatedEOS has no reliable third derivative (cubic Hermite interpolant); use an analytic EOS for thermodynamic_perturbation"))
end

@inline function pressure(eos::TabulatedEOS, x::GrandCanonicalPoint)
    _check_zero_charge(x)
    first(_tabulated_state(eos,x.T))
end

"""
    grandcanonical(eos::TabulatedEOS, target::MicrocanonicalPoint; rtol=4eps())

Invert `ε -> T` without an initial guess. The energy density is strictly
increasing in temperature, so the tabulated node energies bracket the root in
one table cell. Newton iteration on the Hermite interpolant converges inside
that bracket, with bisection as a safeguard. Throws a `DomainError` when `ε`
lies outside the table.
"""
function grandcanonical(eos::TabulatedEOS{U,R}, target::MicrocanonicalPoint;
                        rtol=4*eps(R), maxiter::Integer=100) where {U,R}
    isempty(target.n) || throw(DimensionMismatch("TabulatedEOS is a zero-charge EOS; use MicrocanonicalPoint(ε)"))
    ε = target.energy_density
    energies = eos.energy_density
    first(energies) <= ε <= last(energies) ||
        throw(DomainError(ε,"energy density is outside the table range [$(first(energies)), $(last(energies))]"))
    i = _table_cell(energies,ε)
    lo, hi = eos.temperature[i], eos.temperature[i+1]
    εlo, εhi = energies[i], energies[i+1]
    T = lo + (hi-lo)*(ε-εlo)/(εhi-εlo)
    for _ in 1:maxiter
        p, s, dsdT = _tabulated_state(eos,T)
        residual = -p + T*s - ε
        # Shrink the bracket; ε(T) is increasing on the cell.
        residual > 0 ? (hi = T) : (lo = T)
        step = residual/(T*dsdT)
        candidate = T - step
        if !(lo < candidate < hi)
            candidate = (lo+hi)/2
        end
        converged = abs(candidate-T) <= rtol*T
        T = candidate
        (converged || hi - lo <= rtol*T) && return GrandCanonicalPoint(T)
    end
    p, s, _ = _tabulated_state(eos,T)
    throw(InversionError("iteration limit reached", Int(maxiter), -p + T*s - ε))
end

thermodynamic(eos::TabulatedEOS, x::MicrocanonicalPoint; kwargs...) =
    thermodynamic(eos,grandcanonical(eos,x; kwargs...))
