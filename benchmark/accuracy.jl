# Run from the package root: julia --project=. benchmark/accuracy.jl
using FluidumThermodynamics
using ForwardDiff

function check_accuracy()
    errors = zeros(3)
    setprecision(BigFloat,256) do
        for units in (NaturalGeV(),GeVFm(),MeVFm())
            eos = FluidumEOS(; units)
            p(t) = pressure(eos,GrandCanonicalPoint(t))
            for tGeV in range(0.08,1.0; length=1001)
                t = tGeV/energy_scale(units)
                th = thermodynamic(eos,GrandCanonicalPoint(t))
                tb = BigFloat(t)
                reference = (p(tb),ForwardDiff.derivative(p,tb),
                             ForwardDiff.derivative(x -> ForwardDiff.derivative(p,x),tb))
                values = (th.pressure,th.gradient[1],th.hessian[1,1])
                for i in 1:3
                    errors[i] = max(errors[i],Float64(abs((values[i]-reference[i])/reference[i])))
                end
            end
        end
    end
    println("Maximum relative errors against 256-bit pressure/AD over 1001 temperatures and 3 unit systems:")
    println("pressure = ",errors[1],"; gradient = ",errors[2],"; Hessian = ",errors[3])
    @assert all(errors .< 1e-10)
end

check_accuracy()
