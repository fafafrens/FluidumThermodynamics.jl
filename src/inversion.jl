"""
    InversionError

Failure to invert an EOS. Contains the reason, number of completed Newton
steps, and the infinity norm of the last residual.
"""
struct InversionError{T} <: Exception
    reason::String
    iterations::Int
    residual::T
end

function Base.showerror(io::IO, err::InversionError)
    print(io, "EOS inversion failed: ", err.reason, " after ", err.iterations,
          " Newton steps (residual = ", err.residual, ")")
end

_physical_point(x::GrandCanonicalPoint) = x.T > zero(x.T) && all(isfinite, coordinates(x))
_residual_norm(r) = isempty(r) ? zero(eltype(r)) : norm(r, Inf)

function _trial(evaluate, point, y, domain)
    domain(point(y)) || return nothing
    result = try
        evaluate(y)
    catch err
        # A candidate outside a concrete EOS's domain can be rejected by damping.
        err isa DomainError || rethrow()
        return nothing
    end
    values, jacobian = result
    all(isfinite, values) && all(isfinite, jacobian) || return nothing
    result
end

function _newton(evaluate, point, initial, target;
                 atol, rtol, maxiter, maxbacktracks, domain)
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    maxbacktracks >= 0 || throw(ArgumentError("maxbacktracks must be nonnegative"))
    isfinite(atol) && atol > 0 || throw(ArgumentError("atol must be finite and positive"))
    isfinite(rtol) && rtol >= 0 || throw(ArgumentError("rtol must be finite and nonnegative"))
    all(isfinite, target) || throw(ArgumentError("target values must be finite"))

    y = float.(initial)
    result = _trial(evaluate, point, y, domain)
    isnothing(result) && throw(ArgumentError("initial guess is outside the EOS domain or has nonfinite derivatives"))
    # Each equation has its own scale; energy must not hide a charge residual.
    tolerance = atol .+ rtol .* abs.(target)
    for iteration in 0:maxiter
        values, jacobian = result
        residual = values - target
        norm_residual = _residual_norm(residual)
        score = isempty(residual) ? zero(norm_residual) : norm(residual ./ tolerance, Inf)
        score <= 1 && return point(y)
        iteration == maxiter && throw(InversionError("iteration limit reached", iteration, norm_residual))
        step = try
            jacobian \ residual
        catch err
            err isa SingularException || rethrow()
            throw(InversionError("singular Jacobian", iteration, norm_residual))
        end
        all(isfinite, step) || throw(InversionError("singular or nonfinite Newton step", iteration, norm_residual))

        accepted = false
        damping = one(eltype(y))
        for _ in 0:maxbacktracks
            candidate = y - damping*step
            trial = _trial(evaluate, point, candidate, domain)
            if !isnothing(trial)
                trial_score = norm((trial[1] - target) ./ tolerance, Inf)
                if trial_score <= 1 || trial_score < score
                    y, result = candidate, trial
                    accepted = true
                    break
                end
            end
            damping /= 2
        end
        accepted || throw(InversionError("no admissible step reduces the residual", iteration, norm_residual))
    end
    error("unreachable")
end

"""
    grandcanonical(eos, target::CanonicalPoint, μguess::SVector; kwargs...)

Invert `(T,n) -> (T,μ)` with damped Newton iteration. The Jacobian is the
charge susceptibility matrix. Keyword defaults: `atol=1e-12`,
`rtol=1e-10`, `maxiter=30`, and `maxbacktracks=30`.
Tolerances are applied independently to each equation as
`atol + rtol*abs(target[i])`.

`domain(x)` can restrict admissible grand-canonical points; its default
requires finite coordinates and positive temperature. Throws `InversionError`
on solver failure. Inversion is local and the guess selects the branch.
"""
function grandcanonical(eos::AbstractEquationOfState,
                        target::CanonicalPoint{T,N}, μguess::SVector{N,U};
                        atol=1e-12, rtol=1e-10, maxiter::Integer=30,
                        maxbacktracks::Integer=30, domain=_physical_point) where {T,N,U}
    point(μ) = GrandCanonicalPoint(target.T, μ)
    function evaluate(μ)
        th = thermodynamic(eos, point(μ))
        charge_densities(th), susceptibilities(th)
    end
    _newton(evaluate, point, μguess, target.n;
            atol, rtol, maxiter, maxbacktracks, domain)
end

"""
    grandcanonical(eos, target::MicrocanonicalPoint, guess::GrandCanonicalPoint; kwargs...)

Invert `(ε,n) -> (T,μ)` using only the pressure Hessian. For
`ε = -p + x⋅∇p`, the energy Jacobian row is `(∇²p)*x`; the charge
rows are the corresponding Hessian rows. Uses the same solver keywords,
domain constraints, and failure reporting as canonical inversion.
"""
function grandcanonical(eos::AbstractEquationOfState, target::MicrocanonicalPoint,
                        guess::GrandCanonicalPoint;
                        atol=1e-12, rtol=1e-10, maxiter::Integer=30,
                        maxbacktracks::Integer=30, domain=_physical_point)
    initial = coordinates(guess)
    D = length(initial)
    length(target.n) == D-1 || throw(DimensionMismatch(
        "guess and microcanonical point have different numbers of charges"))
    target_values = SVector(target.energy_density, target.n...)
    function evaluate(y)
        x = grandcanonical_from_coordinates(y)
        th = thermodynamic(eos,x)
        values = SVector(energy_density(x,th), charge_densities(th)...)
        dε = th.hessian*y
        jacobian = SMatrix{D,D}(ntuple(k -> begin
            i = (k-1) % D + 1
            j = (k-1) ÷ D + 1
            i == 1 ? dε[j] : th.hessian[i,j]
        end, D*D))
        values, jacobian
    end
    _newton(evaluate, grandcanonical_from_coordinates, initial, target_values;
            atol, rtol, maxiter, maxbacktracks, domain)
end

"""Alias for `grandcanonical(eos, target, guess; kwargs...)`."""
@inline invert(eos::AbstractEquationOfState,
               target::Union{CanonicalPoint,MicrocanonicalPoint}, guess; kwargs...) =
    grandcanonical(eos,target,guess; kwargs...)
