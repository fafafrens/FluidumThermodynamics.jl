const charm_list_file = joinpath(@__DIR__,"fixtures","particles_charm.data")

@testset "Particle list reader" begin
    table = read_particle_table(charm_list_file)
    @test length(table) == 381
    jpsi = only(filter(r -> r.name == "jp3096zer",table))
    @test jpsi.mass == 3.09687 && jpsi.spin == 1 && jpsi.Nc == 1 && jpsi.Nac == 1 && jpsi.MC == 443
    @test charm_count(jpsi) == 2
    d0 = only(filter(r -> r.name == "Dc1865zer",table))
    @test d0.mass == 1.8646 && d0.Naq == 1 && charm_count(d0) == 1 && d0.MC == 421
    # Commented rows (pentaquarks, Σc) are skipped.
    @test !any(r -> startswith(r.name,"penta") || r.name == "Sc2455ppl",table)

    # Default: the charm hadrons, with charge Nc + Nac and antiparticles counted.
    list = read_particle_list(charm_list_file)
    @test list isa Vector{HadronSpecies{Float64,1}}
    @test length(list) == 9 == count(r -> charm_count(r) > 0,table)
    byname = Dict(r.name => s for (r,s) in zip(filter(r -> charm_count(r) > 0,table),list))
    @test byname["jp3096zer"].charges == SVector(2.0)
    @test byname["jp3096zer"].degeneracy == 3       # self-conjugate: not doubled
    @test byname["Dc1865zer"].charges == SVector(1.0)
    @test byname["Dc1865zer"].degeneracy == 2       # D⁰ and D̄⁰
    @test byname["Lc2286plu"].degeneracy == 4       # spin 1/2, Λc and its antiparticle
    @test all(s -> s.statistics == 0,list)
    plain = read_particle_list(charm_list_file; antiparticles=false)
    @test [s.degeneracy for s in plain] == [2*r.spin + 1 for r in table if charm_count(r) > 0]
    @test [s.mass for s in plain] == [s.mass for s in list]

    # Selections.
    @test length(read_particle_list(charm_list_file; maxmass=3.0)) == 8
    @test all(s -> 1.9 < s.mass < 2.1,read_particle_list(charm_list_file; minmass=1.9,maxmass=2.1))
    @test length(read_particle_list(charm_list_file; condition=r -> charm_count(r) == 1 && r.spin == 0)) == 5
    baryon(r) = (r.Nq + r.Ns + r.Nc - r.Naq - r.Nas - r.Nac)/3
    light = read_particle_list(charm_list_file; charges=r -> (baryon(r),charm_count(r)),
                               condition=_ -> true, antiparticles=false)
    # The massless row is outside the default window minmass = 0 < m.
    @test length(light) == 380 && light isa Vector{HadronSpecies{Float64,2}}
    @test only(filter(s -> s.mass == 2.2864,light)).charges == SVector(1.0,1.0)
    quantum = read_particle_list(charm_list_file; statistics=:quantum)
    @test [s.statistics for s in quantum] == [isinteger(r.spin) ? 1 : -1 for r in table if charm_count(r) > 0]
    @test read_particle_list(charm_list_file; statistics=:fermion)[1].statistics == -1
    @test_throws ArgumentError read_particle_list(charm_list_file; statistics=:classical)
    @test_throws ArgumentError read_particle_list(charm_list_file; minmass=10.0)

    # Formatting: tabs, comments, and blank lines; malformed rows name the line.
    mktemp() do path, io
        write(io,"# header\n\nDc1865zer\t1.8646 0. 0.0 0.5 0.5 0. 0. 1. 0. 1. 0. 421 # D0\n")
        close(io)
        @test only(read_particle_list(path)).mass == 1.8646
    end
    mktemp() do path, io
        write(io,"# header\nDc1865zer 1.8646 0. 0.0 0.5 0.5 0. 0. 1. 0. 1. 0.\n")
        close(io)
        err = try read_particle_table(path); nothing catch e; e end
        @test err isa ArgumentError && occursin(":2:",err.msg)
    end
    mktemp() do path, io
        write(io,"Dc1865zer heavy 0. 0.0 0.5 0.5 0. 0. 1. 0. 1. 0. 421\n")
        close(io)
        @test_throws ArgumentError read_particle_table(path)
    end

    # The HRG of the list: charm density Σqᵢnᵢ of Boltzmann species.
    eos = HadronResonanceGas(list)
    T, μ = 0.155, 0.08
    th = thermodynamic(eos,GrandCanonicalPoint(T,μ))
    bk = FluidumThermodynamics.besselkx
    n(s) = s.degeneracy*s.mass^2*T/(2*pi^2)*exp(s.charges[1]*μ/T - s.mass/T)*bk(2,s.mass/T)
    @test charge_density(th,1) ≈ sum(s -> s.charges[1]*n(s),list)
    @test susceptibilities(th)[1,1] ≈ sum(s -> s.charges[1]^2*n(s),list)/T
end
