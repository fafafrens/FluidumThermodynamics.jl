struct DerivativesOnlyEOS <: AbstractEquationOfState{GeVFm}
    unit_scale::Float64
end
DerivativesOnlyEOS() = DerivativesOnlyEOS(unit_scale(GeVFm()))
function FluidumThermodynamics.thermodynamic(eos::DerivativesOnlyEOS, x::GrandCanonicalPoint{T,0}) where {T}
    ThermodynamicState(eos,x.T^4, SVector(4*x.T^3), SMatrix{1,1}(12*x.T^2))
end

@testset "Pressure derivatives and representations" begin
    for N in 0:3
        χ = SMatrix{N,N,Float64}(Matrix{Float64}(I,N,N))
        eos = PolynomialMultiChargeEOS(3.0, χ)
        μ = SVector{N,Float64}(ntuple(i -> 0.01*i, N))
        x = GrandCanonicalPoint(0.2, μ)
        th = thermodynamic(eos,x)
        @test th.pressure ≈ 3*x.T^4 + x.T^2*dot(μ,χ*μ)/2
        @test entropy_density(th) ≈ 12*x.T^3 + x.T*dot(μ,χ*μ)
        @test charge_densities(th) ≈ x.T^2*χ*μ
        @test susceptibilities(th) ≈ x.T^2*χ
        @test th.hessian[1,1] ≈ 36*x.T^2 + dot(μ,χ*μ)
        @test th.hessian[2:end,1] ≈ 2*x.T*χ*μ
        @test energy_density(x,th) ≈ 3*th.pressure
        @test canonical(eos,x).n == canonical(x,th).n
        @test microcanonical(eos,x).energy_density == microcanonical(x,th).energy_density
        @test eltype(charge_densities(th)) == Float64
        @test eltype(susceptibilities(th)) == Float64
        for i in 1:N
            @test charge_density(th,i) ≈ x.T^2*μ[i]
            @test susceptibility(th,i,i) ≈ x.T^2
        end
    end
    @test GrandCanonicalPoint(1,0.1).T isa Float64
    @test length(CanonicalPoint(0.2).n) == 0
    @test length(MicrocanonicalPoint(0.2).n) == 0
    @test CanonicalPoint(1,0.1).n == SVector(0.1)
    @test MicrocanonicalPoint(1,0.1).n == SVector(0.1)
    th = ThermodynamicState(big"1.0", SVector(2.0), SMatrix{1,1}(3.0))
    @test th.pressure isa BigFloat
    @test eltype(th.gradient) == BigFloat
    @test eltype(th.hessian) == BigFloat
    @test_throws DimensionMismatch thermodynamic(
        PolynomialMultiChargeEOS(3.0, ones(2,3)), GrandCanonicalPoint(0.2,0.0,0.0))
end

@testset "Composition preserves specialized evaluations" begin
    eos = DerivativesOnlyEOS()
    x = GrandCanonicalPoint(0.3)
    reference = thermodynamic(eos,x)
    # No pressure method exists: these operations must use the specialized state.
    for model in (eos+eos, 2*eos, eos*2, eos/0.5, -(-2*eos), 3*eos-eos)
        th = thermodynamic(model,x)
        @test th.pressure ≈ 2*reference.pressure
        @test th.gradient ≈ 2*reference.gradient
        @test th.hessian ≈ 2*reference.hessian
    end
end

@testset "Plain numbers are grand-canonical coordinates" begin
    for eos in (IdealQCD(), PolynomialMultiChargeEOS(1.0,SMatrix{2,2}(1.0,0.1,0.1,2.0)),
                HadronResonanceGas([HadronSpecies(0.14,1,(1,0)),HadronSpecies(0.49,2,(-1,1))]))
        T, μ = 0.2, SVector(0.03,-0.01)
        M = eos isa IdealQCD ? 1 : 2
        μM = SVector{M}(μ[1:M])
        expected = thermodynamic(eos,GrandCanonicalPoint(T,μM))
        for actual in (thermodynamic(eos,T,μM...), @inferred(thermodynamic(eos,T,μM)))
            @test actual.pressure == expected.pressure
            @test actual.gradient == expected.gradient
            @test actual.hessian == expected.hessian
        end
        eos isa PolynomialMultiChargeEOS || @test @inferred(thermodynamic(eos,T)).gradient == thermodynamic(eos,GrandCanonicalPoint(T)).gradient
    end
    @test thermodynamic(IdealQCD(),1//5).pressure ≈ thermodynamic(IdealQCD(),0.2).pressure
    @test isempty(Test.detect_ambiguities(FluidumThermodynamics))
end


@testset "Enthalpy, trace anomaly, and speed of sound" begin
    # A conformal EOS has ε = 3p, so p = ε/3 at any charge density.
    for x in (GrandCanonicalPoint(0.2), GrandCanonicalPoint(0.2,0.1))
        eos = IdealQCD()
        th = thermodynamic(eos,x)
        @test enthalpy(x,th) ≈ energy_density(x,th)+th.pressure
        @test abs(trace_anomaly(x,th)) <= 1e-12*energy_density(x,th)
        dp = @inferred pressure_derivatives(x,th)
        @test dp[1] ≈ 1/3
        @test all(d -> abs(d) <= 1e-12, dp[2:end])
        @test @inferred(speed_of_sound_squared(x,th)) ≈ 1/3
        @test speed_of_sound_squared(eos,x) == speed_of_sound_squared(x,th)
        @test enthalpy(eos,x) == enthalpy(x,th)
    end
    # Without charges, c_s² = s/(T ∂s/∂T).
    eos = FluidumEOS()
    x = GrandCanonicalPoint(0.18)
    th = thermodynamic(eos,x)
    @test speed_of_sound_squared(x,th) ≈ entropy_density(th)/(x.T*th.hessian[1,1])
    @test trace_anomaly(eos,x) ≈ energy_density(x,th)-3*th.pressure
    # Independent check with charges: differentiate (p, ε, s, n) with ForwardDiff
    # and move along the direction that keeps every nᵢ/s fixed.
    gas = HadronResonanceGas([HadronSpecies(0.14,1,(0,1); statistics=:boson),
                              HadronSpecies(0.14,1,(0,-1); statistics=:boson),
                              HadronSpecies(0.94,2,(1,1); statistics=:fermion),
                              HadronSpecies(0.94,2,(-1,-1); statistics=:fermion),
                              HadronSpecies(1.23,4,(1,0); statistics=:fermion)]; terms=40)
    y = SVector(0.15,0.2,0.03)
    x = GrandCanonicalPoint(y[1],y[2],y[3])
    th = thermodynamic(gas,x)
    function observables(v)
        z = GrandCanonicalPoint(v[1],v[2],v[3])
        t = thermodynamic(gas,z)
        [t.pressure, energy_density(z,t), t.gradient[1], t.gradient[2], t.gradient[3]]
    end
    J = ForwardDiff.jacobian(observables,Vector(y))
    dp, dε, ds, dn = J[1,:], J[2,:], J[3,:], J[4:5,:]
    @test J[4:5,:]' * pressure_derivatives(x,th)[2:3] + dε*pressure_derivatives(x,th)[1] ≈ dp rtol=1e-10
    s, n = th.gradient[1], th.gradient[2:3]
    constraints = reduce(vcat, [s*dn[i,:]' - n[i]*ds' for i in 1:2])
    v = nullspace(constraints)[:,1]
    @test speed_of_sound_squared(x,th) ≈ dot(dp,v)/dot(dε,v) rtol=1e-10
    @test 0 < speed_of_sound_squared(x,th) < 1/3
end
