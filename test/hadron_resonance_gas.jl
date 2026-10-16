# Independent momentum integral, differentiated under the integral sign.
# Composite Simpson quadrature needs no production integration dependency.
function hrg_integral(particles, v)
    T = v[1]
    total = zero(T)
    step = 8.0/4000
    for p in particles
        μ = sum(p.charges .* v[2:end])
        for i in 0:4000
            momentum = i*step
            E = sqrt(momentum^2+p.mass^2)
            occupation = inv(exp((E-μ)/T)-p.statistics)
            weight = i == 0 || i == 4000 ? 1 : isodd(i) ? 4 : 2
            total += weight*step/3*p.degeneracy/(6*pi^2)*momentum^4/E*occupation
        end
    end
    total
end

@testset "Hadron resonance gas" begin
    for stats in (:boltzmann,:boson,:fermion)
        particles = [HadronSpecies(0.14,1,(1,0); statistics=stats),
                     HadronSpecies(0.49,2,(-1,1); statistics=stats)]
        eos = HadronResonanceGas(particles; terms=64)
        v = SVector(0.16,0.01,-0.02)
        x = GrandCanonicalPoint(v[1],v[2],v[3])
        th = @inferred thermodynamic(eos,x)
        f(v) = hrg_integral(particles,v)
        @test th.pressure ≈ f(v) rtol=1e-8
        @test th.gradient ≈ ForwardDiff.gradient(f,v) rtol=1e-8
        @test th.hessian ≈ ForwardDiff.hessian(f,v) rtol=1e-8
        @test pressure(eos,x) ≈ th.pressure
        @test issymmetric(th.hessian)
        for units in (GeVFm(),MeVFm())
            factor = 1/energy_scale(units)
            ps = [HadronSpecies(p.mass*factor,p.degeneracy,p.charges; statistics=stats) for p in particles]
            other = HadronResonanceGas(ps; units,terms=64)
            xm = convert_units(units,NaturalGeV(),x)
            actual = thermodynamic(other,xm)
            expected = convert_units(units,th)
            @test actual.pressure ≈ expected.pressure
            @test actual.gradient ≈ expected.gradient
            @test actual.hessian ≈ expected.hessian
            @test unit_system(actual) == units
        end
        zerocharge = thermodynamic(eos,GrandCanonicalPoint(0.16))
        neutral = thermodynamic(eos,GrandCanonicalPoint(0.16,0.0,0.0))
        @test zerocharge.pressure ≈ neutral.pressure
        @test zerocharge.gradient[1] ≈ neutral.gradient[1]
        @test zerocharge.hessian[1] ≈ neutral.hessian[1,1]
        reduced = @inferred thermodynamic(eos,GrandCanonicalPoint(0.16,0.01))
        padded = thermodynamic(eos,GrandCanonicalPoint(0.16,0.01,0.0))
        @test reduced.pressure ≈ padded.pressure
        @test reduced.gradient ≈ padded.gradient[1:2]
        @test reduced.hessian ≈ padded.hessian[1:2,1:2]
        recovered = grandcanonical(eos,microcanonical(x,th),GrandCanonicalPoint(0.15,0.0,0.0))
        @test recovered.T ≈ x.T rtol=1e-7
        @test recovered.μ ≈ x.μ rtol=1e-7
    end
    neutral = HadronResonanceGas([HadronSpecies(0.14,1)])
    @test length(thermodynamic(neutral,GrandCanonicalPoint(0.16)).gradient) == 1
    pair = HadronResonanceGas([HadronSpecies(0.938,2,(1,)),HadronSpecies(0.938,2,(-1,))])
    th = thermodynamic(pair,GrandCanonicalPoint(0.16,0.0))
    @test th.gradient[2] == 0
    @test th.hessian[1,2] == 0
    @test th.hessian[2,2] > 0
    @test thermodynamic(pair,GrandCanonicalPoint(0.001,0.9)).pressure > 0
    @test_throws DimensionMismatch thermodynamic(pair,GrandCanonicalPoint(0.16,0.0,0.0))
    @test_throws DomainError thermodynamic(pair,GrandCanonicalPoint(0.0,0.0))
    quantum = HadronResonanceGas([HadronSpecies(0.14,1,(1,); statistics=:boson)])
    @test_throws DomainError pressure(quantum,GrandCanonicalPoint(0.16,0.15))
    @test_throws ArgumentError HadronSpecies(0.0,1)
    @test_throws ArgumentError HadronSpecies(0.14,1; statistics=:unknown)
    @test_throws ArgumentError HadronResonanceGas([])
    @test_throws ArgumentError HadronResonanceGas(pair.particles; terms=0)
    @test_throws DimensionMismatch HadronResonanceGas([HadronSpecies(0.14,1),HadronSpecies(0.938,2,(1,))])

    @testset "HeavyQuark" begin
        hq = HeavyQuark()
        @test hq isa HeavyQuark
        @test hq.bulk isa FluidumEOS
        species = only(hq.charm.particles)
        @test species.mass == 1.5
        @test species.degeneracy == 6
        @test species.charges == SVector(1.0)
        @test species.statistics == 0
        T,μ = 0.3,0.1
        x = GrandCanonicalPoint(T,μ)
        th = thermodynamic(hq,x)
        bulk = thermodynamic(FluidumEOS(),GrandCanonicalPoint(T))
        charm = thermodynamic(hq.charm,x)
        # Default: one potential p = p_bulk(T) + p_charm(T,μ).
        @test FluidumThermodynamics.backreaction(hq)
        @test th.pressure ≈ bulk.pressure + charm.pressure
        @test pressure(hq,x) ≈ th.pressure
        @test entropy_density(th) ≈ entropy_density(bulk) + entropy_density(charm)
        @test th.hessian[1,1] ≈ bulk.hessian[1,1] + charm.hessian[1,1]
        @test charge_density(th,1) == charge_density(charm,1)
        @test th.hessian[2,2] == charm.hessian[2,2]
        @test th.hessian[1,2] == th.hessian[2,1] == charm.hessian[1,2]
        # The state is the gradient and Hessian of `pressure` (central differences).
        h = 1e-5
        dp(dT,dμ) = (pressure(hq,GrandCanonicalPoint(T+dT,μ+dμ)) - pressure(hq,GrandCanonicalPoint(T-dT,μ-dμ)))/2
        @test dp(h,0.0)/h ≈ entropy_density(th) rtol=1e-6
        @test dp(0.0,h)/h ≈ charge_density(th,1) rtol=1e-6
        dn(dT) = (charge_density(thermodynamic(hq,GrandCanonicalPoint(T+dT,μ)),1) -
                  charge_density(thermodynamic(hq,GrandCanonicalPoint(T-dT,μ)),1))/(2*dT)
        ds(dμ) = (entropy_density(thermodynamic(hq,GrandCanonicalPoint(T,μ+dμ))) -
                  entropy_density(thermodynamic(hq,GrandCanonicalPoint(T,μ-dμ))))/(2*dμ)
        @test dn(h) ≈ th.hessian[1,2] rtol=1e-6
        @test ds(h) ≈ th.hessian[1,2] rtol=1e-6   # Maxwell relation ∂s/∂μ = ∂n/∂T
        z = 1.5/T
        expected = 6*1.5^2*T^2/(2*pi^2)*FluidumThermodynamics.besselkx(2,z)*exp(-z)*exp(μ/T)
        @test pressure(hq.charm,x) ≈ expected
        @test charge_density(th,1) ≈ expected/T
        # Omitted chemical potential means μ = 0, temperature derivatives only.
        th0 = thermodynamic(hq,GrandCanonicalPoint(T))
        charm0 = thermodynamic(hq.charm,GrandCanonicalPoint(T,0.0))
        @test length(th0.gradient) == 1
        @test th0.pressure ≈ bulk.pressure + charm0.pressure
        @test entropy_density(th0) ≈ entropy_density(bulk) + entropy_density(charm0)
        @test pressure(hq,GrandCanonicalPoint(T)) ≈ th0.pressure
        @test_throws DimensionMismatch thermodynamic(hq,GrandCanonicalPoint(T,μ,0.0))
        @test_throws DimensionMismatch pressure(hq,GrandCanonicalPoint(T,μ,0.0))
        # Fluidum convention: the quark gas does not back-react on the bulk.
        pert = HeavyQuark(backreaction=false)
        thp = thermodynamic(pert,x)
        @test !FluidumThermodynamics.backreaction(pert)
        @test thp.pressure == bulk.pressure
        @test pressure(pert,x) ≈ thp.pressure
        @test entropy_density(thp) == entropy_density(bulk)
        @test thp.hessian[1,1] == bulk.hessian[1,1]
        @test charge_density(thp,1) == charge_density(charm,1)
        @test thp.hessian[2,2] == charm.hessian[2,2]
        @test thp.hessian[1,2] == thp.hessian[2,1] == charm.hessian[1,2]
        thp0 = thermodynamic(pert,GrandCanonicalPoint(T))
        @test thp0.pressure == bulk.pressure && thp0.gradient == bulk.gradient && thp0.hessian == bulk.hessian
        # Without back-reaction s is μ-independent while ∂n/∂T is not: no Maxwell relation.
        dsp(dμ) = (entropy_density(thermodynamic(pert,GrandCanonicalPoint(T,μ+dμ))) -
                   entropy_density(thermodynamic(pert,GrandCanonicalPoint(T,μ-dμ))))/(2*dμ)
        @test dsp(h) == 0
        @test thp.hessian[1,2] ≈ dn(h) rtol=1e-6
        @test HeavyQuark(mass=1.3).charm.particles[1].mass == 1.3
        @test HeavyQuark(bulk=IdealQCD()).bulk isa IdealQCD
        gevfm = HeavyQuark(units=FluidumThermodynamics.GeVFm())
        @test FluidumThermodynamics.unit_system(gevfm) === FluidumThermodynamics.GeVFm()
        @test FluidumThermodynamics.unit_system(gevfm.bulk) === FluidumThermodynamics.GeVFm()
        @test FluidumThermodynamics.unit_system(gevfm.charm) === FluidumThermodynamics.GeVFm()
        two = HadronResonanceGas([HadronSpecies(1.5,6,(1,)),HadronSpecies(1.5,6,(-1,))])
        @test_throws ArgumentError HeavyQuark(FluidumEOS(),two)
        @test_throws ArgumentError HeavyQuark(FluidumEOS(units=FluidumThermodynamics.GeVFm()),hq.charm)
    end
end

@testset "Bessel wrapper derivatives" begin
    bk = FluidumThermodynamics.besselkx
    for ν in 0:3, x in (0.5, 7.5, 50.0)
        h = 1e-5*x
        first = (bk(ν,x+h) - bk(ν,x-h))/(2h)
        second = (bk(ν,x+h) - 2*bk(ν,x) + bk(ν,x-h))/h^2
        @test ForwardDiff.derivative(t -> bk(ν,t),x) ≈ first rtol=1e-7
        @test ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> bk(ν,s),t),x) ≈ second rtol=1e-4
    end
    # Closed form for ν = 0: d/dx [e^x K₀(x)] = e^x (K₀ - K₁).
    @test ForwardDiff.derivative(t -> bk(0,t),2.0) ≈ bk(0,2.0) - bk(1,2.0)
    # Differentiating the HRG pressure directly reproduces the analytic state.
    hq = HeavyQuark()
    T, μ = 0.25, 0.1
    th = thermodynamic(hq,GrandCanonicalPoint(T,μ))
    @test ForwardDiff.derivative(t -> pressure(hq,GrandCanonicalPoint(t,μ)),T) ≈ entropy_density(th) rtol=1e-10
    @test ForwardDiff.derivative(m -> pressure(hq,GrandCanonicalPoint(T,m)),μ) ≈ charge_density(th,1) rtol=1e-10
    @test ForwardDiff.derivative(t -> ForwardDiff.derivative(m -> pressure(hq,GrandCanonicalPoint(t,m)),μ),T) ≈ th.hessian[1,2] rtol=1e-10
end
