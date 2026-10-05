# Thermodynamic kernel before the performance changes in this benchmark.
module Baseline
using FluidumThermodynamics: FluiduMEoS, GrandCanonicalPoint, ThermodynamicState, unit_scale, unit_system, energy_scale
using StaticArrays
@inline function _fluidum_fit(eos::FluiduMEoS, T)
    temperature_MeV = 1000*energy_scale(unit_system(eos))*T
    z, w = 154/temperature_MeV, 100/temperature_MeV
    numerator = evalpoly(z,(5.208957878352717,eos.a1,eos.a2,eos.a3,eos.a4))
    denominator = evalpoly(z,(1,eos.b1,eos.b2,eos.b3,eos.b4))
    exponential = exp(-eos.c^2*w-eos.d^2*w^2)
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
    r = n/d
    r1 = (n1-r*d1)/d
    r2 = (n2-r*d2-2*r1*d1)/d
    # Temperature derivatives scaled by T and T², respectively.
    loge1 = eos.c^2*w+2*eos.d^2*w^2
    loge2 = -2*eos.c^2*w-6*eos.d^2*w^2
    f = e*r
    tf1 = e*(loge1*r-z*r1)
    t2f2 = e*((loge1^2+loge2)*r + 2*(1-loge1)*z*r1 + z^2*r2)
    scale = unit_scale(eos)
    p = scale*t^4*f
    s = scale*t^3*(4*f+tf1)
    h = scale*t^2*(12*f+8*tf1+t2f2)
    ThermodynamicState(eos,p,SVector(s),SMatrix{1,1}(h))
end

end
