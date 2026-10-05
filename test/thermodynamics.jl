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
