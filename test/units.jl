@testset "Natural GeV is the default" begin
    for eos in (IdealQCD(),FluidumEOS(),PolynomialMultiChargeEOS(3.0,SMatrix{1,1}(1.0)))
        @test unit_system(eos) === NaturalGeV()
        @test eos.unit_scale == 1.0
    end
    for eos in (IdealQCD(),FluidumEOS())
        x = GrandCanonicalPoint(0.2)
        th = thermodynamic(eos,x)
        @test unit_system(th) === NaturalGeV()
        fm = thermodynamic(convert_units(GeVFm(),eos),x)
        factor = unit_scale(GeVFm())
        @test fm.pressure ≈ factor*th.pressure
        @test fm.gradient ≈ factor*th.gradient
        @test fm.hessian ≈ factor*th.hessian
        @test convert_units(NaturalGeV(),fm).pressure ≈ th.pressure
        mev = thermodynamic(convert_units(MeVFm(),eos),GrandCanonicalPoint(200.0))
        @test mev.pressure ≈ 1000*factor*th.pressure
        @test mev.gradient ≈ factor*th.gradient
        @test mev.hessian ≈ factor*th.hessian/1000
    end
    eos = IdealQCD()
    @test pressure(eos,GrandCanonicalPoint(0.2)) ≈ eos.a1*0.2^4/24
    @test unit_system(ThermodynamicState(1.0,SVector(2.0),SMatrix{1,1}(3.0))) === NaturalGeV()
end

@testset "Stored scale is part of every EOS" begin
    for units in (NaturalGeV(),GeVFm(),MeVFm())
        for eos in (IdealQCD(; units),FluidumEOS(; units),
                    PolynomialMultiChargeEOS(3.0,SMatrix{1,1}(1.0); units))
            @test unit_scale(eos) == eos.unit_scale == unit_scale(units)
            for combined in (eos+eos,2*eos,-eos)
                @test combined.unit_scale == unit_scale(eos)
            end
        end
    end
    converted = convert_units(MeVFm(),IdealQCD(units=GeVFm()))
    @test converted.unit_scale == unit_scale(MeVFm())
    @test converted.unit_scale ≈ unit_scale(IdealQCD(units=GeVFm()))/1e9
end

@testset "EOS and state share their unit convention" begin
    gev = IdealQCD(units=GeVFm())
    mev = IdealQCD(units=MeVFm())
    @test gev isa AbstractEquationOfState{GeVFm}
    @test mev isa AbstractEquationOfState{MeVFm}
    for x in (GrandCanonicalPoint(0.2), GrandCanonicalPoint(0.2,0.03))
        xm = convert_units(MeVFm(),GeVFm(),x)
        th = @inferred thermodynamic(gev,x)
        thm = @inferred thermodynamic(mev,xm)
        @test unit_system(th) === unit_system(gev) === GeVFm()
        @test unit_system(thm) === unit_system(mev) === MeVFm()
        @test thm isa ThermodynamicState{MeVFm}
        @test thm.pressure ≈ 1000*th.pressure
        @test thm.gradient ≈ th.gradient
        @test thm.hessian ≈ th.hessian/1000
        @test energy_density(xm,thm) ≈ 1000*energy_density(x,th)
        converted = @inferred convert_units(MeVFm(),th)
        @test converted.pressure ≈ thm.pressure
        @test converted.gradient ≈ thm.gradient
        @test converted.hessian ≈ thm.hessian
        @test convert_units(GeVFm(),converted).hessian ≈ th.hessian
        @test convert_units(GeVFm(),th) === th
    end
    y = SVector(200.0,30.0)
    f(z) = pressure(mev,GrandCanonicalPoint(z[1],z[2]))
    th = thermodynamic(mev,GrandCanonicalPoint(y[1],y[2]))
    @test th.gradient ≈ ForwardDiff.gradient(f,y)
    @test th.hessian ≈ ForwardDiff.hessian(f,y)
end

@testset "Conversions preserve generic and specialized EOS models" begin
    χ = @SMatrix [2.0 0.3; 0.3 1.5]
    eos = PolynomialMultiChargeEOS(3.0,χ; units=GeVFm())
    x = GrandCanonicalPoint(0.2,0.03,-0.01)
    xm = convert_units(MeVFm(),GeVFm(),x)
    converted = @inferred convert_units(MeVFm(),eos)
    native = PolynomialMultiChargeEOS(3e-9,1e-9*χ; units=MeVFm())
    thm = @inferred thermodynamic(converted,xm)
    expected = thermodynamic(native,xm)
    @test unit_system(converted) === unit_system(thm) === MeVFm()
    @test pressure(converted,xm) ≈ pressure(native,xm)
    @test thm.pressure ≈ expected.pressure
    @test thm.gradient ≈ expected.gradient
    @test thm.hessian ≈ expected.hessian
    @test convert_units(GeVFm(),converted) === eos
    @test convert_units(MeVFm(),converted) === converted
    @test convert_units(GeVFm(),eos) === eos

    # Preserve a specialized evaluation that has no pressure/AD method.
    special = convert_units(MeVFm(),DerivativesOnlyEOS())
    th = @inferred thermodynamic(special,GrandCanonicalPoint(200.0))
    @test unit_system(th) === MeVFm()
    @test th.pressure ≈ 1000*0.2^4
    @test th.gradient[1] ≈ 4*0.2^3
    @test th.hessian[1,1] ≈ 12*0.2^2/1000
end

@testset "Unit conventions survive composition and inversion" begin
    eos = IdealQCD(units=MeVFm())
    x = GrandCanonicalPoint(200.0,30.0)
    th = thermodynamic(eos,x)
    for model in (eos+eos,2*eos,eos*2,eos/0.5,-(-2*eos),3*eos-eos)
        combined = @inferred thermodynamic(model,x)
        @test unit_system(model) === unit_system(combined) === MeVFm()
        @test combined.pressure ≈ 2*th.pressure
        @test combined.hessian ≈ 2*th.hessian
    end
    @test_throws ArgumentError eos+IdealQCD(units=GeVFm())
    @test_throws ArgumentError eos-IdealQCD(units=GeVFm())
    @test thermodynamic(eos+convert_units(MeVFm(),IdealQCD(units=GeVFm())),x).pressure ≈ 2*th.pressure
    original = IdealQCD(units=GeVFm())
    xg = GrandCanonicalPoint(0.2,0.03)
    can = convert_units(MeVFm(),GeVFm(),canonical(original,xg))
    mic = convert_units(MeVFm(),GeVFm(),microcanonical(original,xg))
    @test can.T ≈ x.T
    @test can.n ≈ charge_densities(th)
    @test mic.energy_density ≈ energy_density(x,th)
    @test mic.n ≈ charge_densities(th)
    @test grandcanonical(eos,can,SVector(0.0)).μ ≈ x.μ
    recovered = grandcanonical(eos,mic,GrandCanonicalPoint(180.0,0.0))
    @test recovered.T ≈ x.T
    @test recovered.μ ≈ x.μ
end
