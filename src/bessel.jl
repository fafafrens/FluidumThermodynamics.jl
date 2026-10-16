# Scaled modified Bessel function of the second kind, e^x K_ν(x), from Bessels.jl.
# The wrapper is owned by this package, so the ForwardDiff rule below is not
# type piracy on Bessels or ForwardDiff.
@inline besselkx(ν, x) = Bessels.besselkx(ν, x)

# d/dx [e^x K_ν(x)] = (1 - ν/x) e^x K_ν(x) - e^x K_{ν-1}(x), with K_{-ν} = K_ν.
# Calling the wrapper on the value keeps nested duals (second derivatives) working.
function besselkx(ν, d::ForwardDiff.Dual{T}) where {T}
    x = ForwardDiff.value(d)
    k = besselkx(ν, x)
    derivative = muladd(-ν/x, k, k) - besselkx(abs(ν-1), x)
    ForwardDiff.Dual{T}(k, derivative*ForwardDiff.partials(d))
end
