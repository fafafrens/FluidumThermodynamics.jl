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

const canonical_charm_file = joinpath(@__DIR__,"fixtures","particles_charm.dat")
const canonical_light_file = joinpath(@__DIR__,"fixtures","particles_light_qm_excerpt.dat")

@testset "Canonical FastReso particle list" begin
    table = read_particle_table(canonical_charm_file)
    @test length(table) == 91
    row(id) = only(filter(r -> r.id == id,table))
    jpsi = row(443)
    @test jpsi.name == "J/psi(1S)" && jpsi.mass == 3.0969 && jpsi.degeneracy == 3 && jpsi.C == 0
    # The PDG digits nq1 nq2 nq3 give the charm quarks and antiquarks.
    @test FluidumThermodynamics._pdg_charm_quarks(443) == 2
    @test FluidumThermodynamics._pdg_charm_quarks(9000443) == 2
    @test FluidumThermodynamics._pdg_charm_quarks(-421) == 1
    @test FluidumThermodynamics._pdg_charm_quarks(104324) == 1
    @test FluidumThermodynamics._pdg_charm_quarks(211) == 0
    # c + c̄ (default) and net charm.
    for (id,q,qnet) in ((421,1,1),(-421,1,-1),(4122,1,1),(-4122,1,-1),(-433,1,-1),
                        (443,2,0),(441,2,0),(9000443,2,0),(100443,2,0))
        @test charm_count(row(id)) == q
        @test charm_count(row(id); net=true) == qnet == row(id).C
    end
    @test count(r -> charm_count(r) == 1,table) == 78
    @test count(r -> charm_count(r) == 2,table) == 13

    # Default list: every row, degeneracies from the file, antiparticles not doubled.
    list = read_particle_list(canonical_charm_file)
    @test list isa Vector{HadronSpecies{Float64,1}} && length(list) == 91
    @test [s.degeneracy for s in list] == [r.degeneracy for r in table]
    @test [s.charges[1] for s in list] == [charm_count(r) for r in table]
    @test list[findfirst(r -> r.id == 423,table)].degeneracy == 3   # D*(2007)⁰
    @test_throws ArgumentError read_particle_list(canonical_charm_file; antiparticles=true)
    @test read_particle_list(canonical_charm_file; antiparticles=false) == list
    quantum = read_particle_list(canonical_charm_file; statistics=:quantum)
    @test quantum[findfirst(r -> r.id == 4122,table)].statistics == -1   # Λc fermion
    @test quantum[findfirst(r -> r.id == 421,table)].statistics == 1     # D⁰ boson

    # Net charm: charmonia drop out and the gas is charge-symmetric.
    netlist = read_particle_list(canonical_charm_file; net=true)
    @test length(netlist) == 78
    @test sort([s.charges[1] for s in netlist]) == [fill(-1.0,39); fill(1.0,39)]
    eos = HadronResonanceGas(netlist)
    T = 0.155
    @test abs(charge_density(thermodynamic(eos,GrandCanonicalPoint(T,0.0)),1)) < 1e-14
    @test charge_density(thermodynamic(eos,GrandCanonicalPoint(T,0.03)),1) ≈
          -charge_density(thermodynamic(eos,GrandCanonicalPoint(T,-0.03)),1)

    # Quark-model ids are not PDG numbers: their digits must not count as charm.
    light = read_particle_table(canonical_light_file)
    @test length(light) == 10
    @test any(r -> FluidumThermodynamics._pdg_charm_quarks(r.id) > 0,light)   # 5000440, 2248, ...
    @test all(r -> charm_count(r) == 0 && charm_count(r; net=true) == 0,light)
    @test_throws ArgumentError read_particle_list(canonical_light_file)
    baryons = read_particle_list(canonical_light_file; charges=r -> (r.B,),condition=_ -> true,
                                 statistics=:quantum)
    @test [s.charges[1] for s in baryons] == [r.B for r in light]
    @test [s.statistics for s in baryons] == [iszero(r.B) ? 1 : -1 for r in light]

    # Format detection, and forcing the wrong format.
    @test eltype(read_particle_table(charm_list_file)) !== eltype(table)
    @test read_particle_table(canonical_charm_file; format=:canonical) == table
    @test_throws ArgumentError read_particle_table(canonical_charm_file; format=:therminator)
    @test_throws ArgumentError read_particle_table(charm_list_file; format=:canonical)
    @test_throws ArgumentError read_particle_table(canonical_charm_file; format=:pdg)
    mktemp() do path, io
        write(io,"# id name mass width degeneracy B S C Bot I Q [aliases...]\n",
                 "421 D0 1.86484 0.0 1.0 0 0 1 0 0.5 0 D^0 Dzero\n\n421 D0 1.86484 0.0 1.0 0 0 1\n")
        close(io)
        err = try read_particle_table(path); nothing catch e; e end
        @test err isa ArgumentError && occursin(":4:",err.msg)
    end
    mktemp() do path, io
        write(io,"421 D0 1.86484 0.0 1.0 0 0 1 0 0.5 0 D^0 Dzero\n")
        close(io)
        @test only(read_particle_list(path)).mass == 1.86484   # aliases ignored
    end

    # THERMINATOR lists: net charm is Nc - Nac.
    old = read_particle_table(charm_list_file)
    @test charm_count(only(filter(r -> r.name == "jp3096zer",old)); net=true) == 0
    oldnet = read_particle_list(charm_list_file; net=true, antiparticles=false)
    @test length(oldnet) == 8 && all(s -> s.charges[1] == 1,oldnet)
end
