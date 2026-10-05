@testset "IdealQCD evaluation from other representations" begin
    for units in (NaturalGeV(),GeVFm(),MeVFm())
        eos = IdealQCD(; units)
        e = energy_scale(units)
        for μ in (-0.1,0.0,1e-10,0.15)
            x = GrandCanonicalPoint(0.2/e,μ/e)
            th = thermodynamic(eos,x)
            can = canonical(x,th)
            mic = microcanonical(x,th)
            recovered = grandcanonical(eos,can)
            @test recovered.μ ≈ x.μ atol=1e-13
            tc = @inferred thermodynamic(eos,can)
            tm = thermodynamic(eos,mic; atol=1e-15)
            for state in (tc,tm)
                @test unit_system(state) === units
                @test state.pressure ≈ th.pressure
                @test state.gradient ≈ th.gradient
                @test state.hessian ≈ th.hessian
            end
        end
        x = GrandCanonicalPoint(0.2/e)
        th = thermodynamic(eos,x)
        @test thermodynamic(eos,canonical(x,th)).hessian ≈ th.hessian
        @test thermodynamic(eos,microcanonical(x,th)).hessian ≈ th.hessian
    end

    eos = IdealQCD()
    # The analytic cubic remains accurate in both small- and large-μ/T regimes.
    for μ in (1e-14,10.0,-10.0)
        x = GrandCanonicalPoint(0.2,μ)
        @test grandcanonical(eos,canonical(eos,x)).μ[1] ≈ μ rtol=1e-12
    end
    @test grandcanonical(eos,canonical(eos,GrandCanonicalPoint(0.0,0.3))).μ[1] ≈ 0.3
    pure_glue = IdealQCD(3,0)
    @test grandcanonical(pure_glue,CanonicalPoint(0.2,0.0)).μ[1] == 0
    @test_throws DomainError grandcanonical(pure_glue,CanonicalPoint(0.2,0.1))
    @test_throws DimensionMismatch thermodynamic(eos,CanonicalPoint(0.2,0.0,0.0))
    @test_throws DimensionMismatch thermodynamic(eos,MicrocanonicalPoint(0.1,0.0,0.0))
    target = microcanonical(eos,GrandCanonicalPoint(0.2,0.1))
    @test thermodynamic(eos,target; guess=GrandCanonicalPoint(0.18,0.0)).pressure ≈ target.energy_density/3
    @test_throws InversionError thermodynamic(eos,target; maxiter=0)
    # Differentiate the analytic canonical inverse using dμ/dn = 1/χ.
    x = GrandCanonicalPoint(0.2,0.03)
    th = thermodynamic(eos,x)
    n = charge_densities(th)[1]
    μofn(n) = grandcanonical(eos,CanonicalPoint(x.T,n)).μ[1]
    @test ForwardDiff.derivative(μofn,n) ≈ inv(susceptibility(th,1,1))
end

@testset "Inversion with damping and failure reporting" begin
    eos = IdealQCD()
    for μ in (-0.1,0.0,0.15)
        x = GrandCanonicalPoint(0.2,μ)
        can = canonical(eos,x)
        gc = invert(eos,can,SVector(0.0))
        @test gc.T ≈ x.T
        @test gc.μ ≈ x.μ atol=1e-10
        mic = microcanonical(eos,x)
        gc = invert(eos,mic,GrandCanonicalPoint(0.05,0.0))
        @test gc.T ≈ x.T atol=1e-10
        @test gc.μ ≈ x.μ atol=1e-10
    end
    x0 = GrandCanonicalPoint(0.2)
    @test isempty(grandcanonical(eos,canonical(eos,x0),SVector{0,Float64}()).μ)
    @test grandcanonical(eos,microcanonical(eos,x0),GrandCanonicalPoint(0.01)).T ≈ x0.T
    bounded(x) = 0 < x.T < 0.3 && all(isfinite,x.μ)
    @test grandcanonical(eos,microcanonical(eos,x0),GrandCanonicalPoint(0.01);
                         domain=bounded).T ≈ x0.T
    # An energy scale must not hide a small but nonzero charge target.
    xsmall = GrandCanonicalPoint(0.5,1e-6)
    recovered = grandcanonical(eos,microcanonical(eos,xsmall),GrandCanonicalPoint(0.4,0.0))
    @test recovered.μ ≈ xsmall.μ atol=1e-10
    @test_throws ArgumentError grandcanonical(eos,microcanonical(eos,x0),GrandCanonicalPoint(-0.1))
    @test_throws ArgumentError grandcanonical(eos,CanonicalPoint(0.2,NaN),SVector(0.0))
    @test_throws ArgumentError grandcanonical(eos,CanonicalPoint(0.2,0.1),SVector(0.0); atol=0)
    @test_throws ArgumentError grandcanonical(eos,CanonicalPoint(0.2,0.1),SVector(0.0); maxiter=-1)
    @test_throws DimensionMismatch grandcanonical(eos,MicrocanonicalPoint(0.1,0.0),x0)
    @test_throws InversionError grandcanonical(eos,microcanonical(eos,x0),GrandCanonicalPoint(0.01); maxiter=0)
    @test_throws InversionError grandcanonical(eos,microcanonical(eos,x0),GrandCanonicalPoint(0.01);
                                             domain=bounded,maxbacktracks=0)
    singular = PolynomialMultiChargeEOS(3.0, SMatrix{1,1}(0.0))
    err = try
        grandcanonical(singular,CanonicalPoint(0.2,0.1),SVector(0.0))
        nothing
    catch exception
        exception
    end
    @test err isa InversionError
    @test err.iterations == 0
    @test occursin("singular", sprint(showerror,err))
end
