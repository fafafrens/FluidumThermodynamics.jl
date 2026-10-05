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
end
