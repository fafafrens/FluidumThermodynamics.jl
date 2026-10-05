module LegacyIdealQCD
abstract type EquationOfState end
const fmGeV3 = (1/0.1973261)^3
struct Thermodynamic{T,N,M}
    pressure::T
    gradient::NTuple{N,T}
    hessian::NTuple{M,T}
end
include("fixtures/fluidum_ideal_qcd.jl")
end

module LegacyFluidumFit
abstract type EquationOfState end
const fmGeV3 = (1/0.1973261)^3
struct Thermodynamic{P,G,H}
    pressure::P
    gradient::G
    hessian::H
end
include("fixtures/fluidum_fit.jl")
end

@testset "FluiduMEoS port, derivatives, and units" begin
    eos = FluidumEOS(units=GeVFm())
    mev = FluiduMEoS(units=MeVFm())
    legacy = LegacyFluidumFit.FluiduMEoS()
    @test FluidumEOS === FluiduMEoS
    for t in (0.08,0.12,0.154,0.2,0.5,1.0)
        x = GrandCanonicalPoint(t)
        th = @inferred thermodynamic(eos,x)
        reference = LegacyFluidumFit.thermodynamic(t,legacy)
        @test th.pressure ≈ reference.pressure rtol=1e-10
        @test th.gradient[1] ≈ reference.gradient[1] rtol=1e-10
        @test th.hessian[1,1] ≈ reference.hessian[1] rtol=1e-10
        @test pressure(eos,x) ≈ th.pressure
        p(v) = pressure(eos,GrandCanonicalPoint(v))
        @test ForwardDiff.derivative(p,t) ≈ th.gradient[1] rtol=1e-10
        @test ForwardDiff.derivative(v -> ForwardDiff.derivative(p,v),t) ≈ th.hessian[1,1] rtol=1e-10
        thm = @inferred thermodynamic(mev,GrandCanonicalPoint(1000*t))
        @test unit_system(thm) === MeVFm()
        @test thm.pressure ≈ 1000*th.pressure
        @test thm.gradient ≈ th.gradient
        @test thm.hessian ≈ th.hessian/1000
    end
    x = GrandCanonicalPoint(0.2)
    th = thermodynamic(eos,x)
    @test unit_system(th) === GeVFm()
    @test isempty(charge_densities(th))
    @test thermodynamic(2*eos,x).hessian ≈ 2*th.hessian
    @test thermodynamic(convert_units(MeVFm(),eos),GrandCanonicalPoint(200.0)).hessian ≈ th.hessian/1000
    recovered = grandcanonical(eos,microcanonical(x,th),GrandCanonicalPoint(0.18))
    @test recovered.T ≈ x.T
    @test_throws DimensionMismatch thermodynamic(eos,GrandCanonicalPoint(0.2,0.0))
end
@testset "IdealQCD agrees with Fluidum reference" begin
    for (Nc,Nf) in ((3,2),(3,3),(2,1)), T in (0.1,0.2,0.5), μ in (-0.1,0.0,0.15)
        eos = IdealQCD(Nc,Nf; units=GeVFm())
        old = LegacyIdealQCD.IdealQCD(Nc,Nf)
        th = thermodynamic(eos,GrandCanonicalPoint(T,μ))
        reference = LegacyIdealQCD.thermodynamic(T,μ,old)
        @test th.pressure ≈ reference.pressure
        @test th.gradient ≈ SVector(reference.gradient)
        hTT,hTμ,hμμ = reference.hessian
        @test th.hessian ≈ SMatrix{2,2}(hTT,hTμ,hTμ,hμμ)
        # Also independently check the analytic implementation against AD.
        y = SVector(T,μ)
        f(z) = pressure(eos,GrandCanonicalPoint(z[1],z[2]))
        @test th.gradient ≈ ForwardDiff.gradient(f,y)
        @test th.hessian ≈ ForwardDiff.hessian(f,y)
        @test energy_density(GrandCanonicalPoint(T,μ),th) ≈ 3*th.pressure

        zerocharge = thermodynamic(eos,GrandCanonicalPoint(T))
        reference0 = LegacyIdealQCD.thermodynamic(T,old)
        @test zerocharge.pressure ≈ reference0.pressure
        @test zerocharge.gradient ≈ SVector(reference0.gradient)
        @test zerocharge.hessian ≈ SMatrix{1,1}(reference0.hessian)
    end
    @test_throws ArgumentError IdealQCD(0,2)
    @test_throws ArgumentError IdealQCD(3,-1)
    @test_throws DimensionMismatch pressure(IdealQCD(),GrandCanonicalPoint(0.2,0.0,0.0))
end
