# Conserved → primitive recovery from the example, for several EOS.
include(joinpath(@__DIR__,"..","examples","primitive_recovery.jl"))

function recovery_allocations(eos, N, Mx, My, E)
    primitive_from_conserved(eos, N, Mx, My, E; T0=0.2, μ0=0.05)
    @allocated primitive_from_conserved(eos, N, Mx, My, E; T0=0.2, μ0=0.05)
end

@testset "Conserved to primitive recovery" begin
    for eos in (IdealQCD(), HeavyQuark(), HeavyQuark(backreaction=false),
                PolynomialMultiChargeEOS(3.0,SMatrix{1,1}(2.0)))
        for (T, μ, vx, vy) in ((0.25,0.08,0.3,-0.2),(0.16,0.0,0.0,0.0),(0.4,0.3,0.8,0.1))
            N, Mx, My, E = conserved_from_primitive(eos, T, μ, vx, vy)
            prim, it = @inferred primitive_from_conserved(eos, N, Mx, My, E; T0=0.15, μ0=0.0)
            @test prim ≈ SVector(T, μ, vx, vy) rtol=1e-9 atol=1e-12
            @test it < 50
            # Round trip of the conserved variables.
            @test collect(conserved_from_primitive(eos, prim...)) ≈ [N, Mx, My, E] rtol=1e-9
        end
        N, Mx, My, E = conserved_from_primitive(eos, 0.25, 0.08, 0.3, -0.2)
        @test recovery_allocations(eos, N, Mx, My, E) == 0
        # The analytic Jacobian is the derivative of the residual whenever the
        # state derives from a potential. Without back-reaction ∂s/∂μ = 0 while
        # the Hessian entry is ∂n/∂T, so the Jacobian is only approximate there
        # (Newton still converges, as the round trips above show).
        M2 = Mx^2 + My^2
        f(u) = residual_jacobian(eos, u[1], u[2], E, M2, N)[1]
        u = SVector(0.22, 0.05)
        J, Jad = residual_jacobian(eos, u[1], u[2], E, M2, N)[2], ForwardDiff.jacobian(f, u)
        if eos isa HeavyQuark && !FluidumThermodynamics.backreaction(eos)
            @test !isapprox(J, Jad; rtol=1e-8)
        else
            @test J ≈ Jad rtol=1e-8
        end
    end
    @test_throws DomainError conserved_from_primitive(IdealQCD(), 0.2, 0.0, 0.8, 0.7)
end
