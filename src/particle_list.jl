const _particle_columns = (:name,:mass,:gamma,:spin,:isospin,:I3,:Nq,:Ns,:Naq,:Nas,:Nc,:Nac,:MC)

"""
    read_particle_table(file)

Rows of a FastReso particle list (`particles.data` format), as NamedTuples with
the columns

    name  mass  gamma  spin  isospin  I3  Nq  Ns  Naq  Nas  Nc  Nac  MC

Mass and width are in GeV. `Nq`, `Ns`, `Nc` count light, strange, and charm
quarks and `Naq`, `Nas`, `Nac` the antiquarks. Text after `#` and blank lines
are skipped. Fields may be separated by spaces or tabs.
"""
function read_particle_table(file::AbstractString)
    rows = NamedTuple{_particle_columns,Tuple{String,ntuple(_ -> Float64,11)...,Int}}[]
    for (number,line) in enumerate(eachline(file))
        fields = split(first(split(line,'#';limit=2)))
        isempty(fields) && continue
        length(fields) == 13 || throw(ArgumentError(
            "$file:$number: expected 13 columns, found $(length(fields))"))
        values = ntuple(i -> tryparse(Float64,fields[i+1]),11)
        mc = tryparse(Int,fields[13])
        (any(isnothing,values) || isnothing(mc)) && throw(ArgumentError(
            "$file:$number: cannot parse numeric columns of $(fields[1])"))
        push!(rows,NamedTuple{_particle_columns}((String(fields[1]),values...,mc)))
    end
    rows
end

"""
    charm_count(row)

Number of charm quarks and antiquarks, `Nc + Nac`, of a `read_particle_table`
row: 1 for open charm hadrons, 2 for J/ψ.
"""
charm_count(row) = row.Nc + row.Nac

_self_conjugate(row) = row.Nq == row.Naq && row.Ns == row.Nas && row.Nc == row.Nac && iszero(row.I3)

"""
    read_particle_list(file; charges=row -> (charm_count(row),), condition=<any charge ≠ 0>,
                       minmass=0, maxmass=Inf, statistics=:boltzmann, antiparticles=true)

`HadronSpecies` from a FastReso particle list, in the spirit of Fluidum's
`readresonancelist`. `charges(row)` gives the charges of each species from its
`read_particle_table` row; the default is the heavy-quark count
`Nc + Nac` (`charm_count`). Rows are kept if `minmass < mass < maxmass` and
`condition(row)` holds; by default rows whose charges all vanish are dropped,
so reading a full list yields only the charm hadrons.

Each row is one isospin state with degeneracy `2·spin + 1`. FastReso lists do
not contain the antiparticles: with `antiparticles=true` the degeneracy is
doubled for every row that is not its own conjugate (`Nq = Naq`, `Ns = Nas`,
`Nc = Nac`, `I3 = 0`, e.g. J/ψ), so the conjugate shares the charges of the
row. This is right for counts such as `Nc + Nac` that are the same for
particle and antiparticle; pass `antiparticles=false` and list the
conjugates explicitly for charges that change sign.

`statistics` is `:boltzmann`, `:boson`, `:fermion`, or `:quantum`, which picks
Bose-Einstein or Fermi-Dirac from the spin. Masses are in GeV, so the list
fits an EOS whose energy unit is GeV.
"""
function read_particle_list(file::AbstractString; charges=row -> (charm_count(row),),
                            condition=row -> any(!iszero,charges(row)),
                            minmass::Real=0, maxmass::Real=Inf,
                            statistics::Symbol=:boltzmann, antiparticles::Bool=true)
    statistics in (:boltzmann,:boson,:fermion,:quantum) ||
        throw(ArgumentError("unknown statistics: $statistics"))
    rows = filter(row -> minmass < row.mass < maxmass && condition(row),read_particle_table(file))
    isempty(rows) && throw(ArgumentError("no particle of $file passes the selection"))
    N = length(charges(first(rows)))
    map(rows) do row
        q = charges(row)
        length(q) == N || throw(DimensionMismatch("charges must return $N values for every row"))
        g = (2*row.spin + 1)*(antiparticles && !_self_conjugate(row) ? 2 : 1)
        stat = statistics !== :quantum ? statistics : isinteger(row.spin) ? :boson : :fermion
        HadronSpecies(row.mass,g,SVector{N,Float64}(Tuple(q)); statistics=stat)
    end
end
