using BenchmarkTools
using FluidumThermodynamics
using StaticArrays
using Printf

include("baseline.jl")
include("legacy_fluidum.jl")

@inline current(eos,t) = thermodynamic(eos,GrandCanonicalPoint(t))
@inline baseline(eos,t) = Baseline.thermodynamic(eos,GrandCanonicalPoint(t))
@inline legacy(eos,t) = LegacyFluidum.thermodynamic(t,eos)

function sweep!(out, temperatures, eos, evaluate::F) where {F}
    @inbounds for i in eachindex(temperatures,out)
        th = evaluate(eos,temperatures[i])
        out[i] = SVector(th.pressure,th.gradient[1],th.hessian[1,1])
    end
    out
end

# The legacy state stores its one-entry Hessian in a tuple.
@inline function legacy_state(eos,t)
    th = legacy(eos,t)
    (pressure=th.pressure, gradient=th.gradient, hessian=SMatrix{1,1}(th.hessian))
end

function report(label, trial; count=1)
    best, typical = minimum(trial), median(trial)
    @printf("%-28s %9.2f %9.2f %9d %7d\n",
            label,best.time/count,typical.time/count,best.memory,best.allocs)
end

function main()
    println("Julia ",VERSION,"; CPU target ",Sys.CPU_NAME,"; threads ",Threads.nthreads())
    println("Times in ns/evaluation; bytes and allocations per complete sample.")
    println("Scalar inputs dereferenced through Ref; evals=1000 (sweeps=1); compilation excluded.")
    @printf("%-28s %9s %9s %9s %7s\n","Case","minimum","median","bytes","allocs")
    seconds = parse(Float64,get(ENV,"FLUIDUM_BENCH_SECONDS","1.0"))
    en, ef, em = FluidumEOS(), FluidumEOS(units=GeVFm()), FluidumEOS(units=MeVFm())
    original = LegacyFluidum.FluiduMEoS()
    for t in (0.12,0.2,0.5)
        for (label,f,eos) in (("baseline GeV/fm³",baseline,ef),
                              ("current GeV/fm³",current,ef),
                              ("original Fluidum",legacy,original))
            trial = @benchmark $f($(Ref(eos))[],$(Ref(t))[]) seconds=seconds evals=1000
            report("$label T=$t",trial)
        end
    end
    for (label,eos,t) in (("current GeV⁴",en,0.2),("current MeV/fm³",em,200.0))
        trial = @benchmark current($(Ref(eos))[],$(Ref(t))[]) seconds=seconds evals=1000
        report(label,trial)
    end

    temperatures = collect(range(0.08,1.0; length=4096))
    out = Vector{SVector{3,Float64}}(undef,length(temperatures))
    for (label,f,eos) in (("sweep baseline GeV/fm³",baseline,ef),
                          ("sweep current GeV/fm³",current,ef),
                          ("sweep original Fluidum",legacy_state,original),
                          ("sweep current GeV⁴",current,en))
        trial = @benchmark sweep!($out,$temperatures,$eos,$f) seconds=seconds evals=1
        report(label,trial; count=length(temperatures))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
