const _therminator_columns = (:name,:mass,:gamma,:spin,:isospin,:I3,:Nq,:Ns,:Naq,:Nas,:Nc,:Nac,:MC)
const _canonical_columns = (:id,:name,:mass,:width,:degeneracy,:B,:S,:C,:Bot,:I,:Q)

const _TherminatorRow = NamedTuple{_therminator_columns,Tuple{String,ntuple(_ -> Float64,11)...,Int}}
const _CanonicalRow = NamedTuple{_canonical_columns,Tuple{Int,String,ntuple(_ -> Float64,9)...}}

"""
    read_particle_table(file; format=:auto)

Rows of a FastReso particle list, as NamedTuples. Two formats are read:

- `:canonical`, the FastReso2 tables (`particles_*.dat`), one row per state
  including antiparticles:

      id  name  mass  width  degeneracy  B  S  C  Bot  I  Q  [aliases...]

  `id` is the PDG Monte-Carlo id (negative for antiparticles), `B, S, C, Bot`
  the net baryon number, strangeness, charm, and bottomness, `I` the isospin,
  and `Q` the electric charge. Trailing aliases are ignored.
- `:therminator`, the 13-column `particles.data` layout of FastReso and
  THERMINATOR, without antiparticles:

      name  mass  gamma  spin  isospin  I3  Nq  Ns  Naq  Nas  Nc  Nac  MC

  `Nq`, `Ns`, `Nc` count light, strange, and charm quarks and `Naq`, `Nas`,
  `Nac` the antiquarks.

`format=:auto` picks `:canonical` when the first field of the first row is an
integer. Masses and widths are in GeV. Text after `#` and blank lines are
skipped; fields may be separated by spaces or tabs.
"""
function read_particle_table(file::AbstractString; format::Symbol=:auto)
    format in (:auto,:canonical,:therminator) || throw(ArgumentError("unknown particle-list format: $format"))
    lines = [(number,fields) for (number,fields) in
             enumerate(split(first(split(line,'#';limit=2))) for line in eachline(file)) if !isempty(fields)]
    if format === :auto
        format = !isempty(lines) && !isnothing(tryparse(Int,lines[1][2][1])) ? :canonical : :therminator
    end
    format === :canonical ? _CanonicalRow[_canonical_row(file,n,f) for (n,f) in lines] :
                            _TherminatorRow[_therminator_row(file,n,f) for (n,f) in lines]
end

_unparsable(file, number, fields) =
    ArgumentError("$file:$number: cannot parse numeric columns of $(join(fields[1:min(2,end)],' '))")

function _therminator_row(file, number, fields)
    length(fields) == 13 || throw(ArgumentError(
        "$file:$number: expected 13 columns (THERMINATOR format), found $(length(fields))"))
    values = ntuple(i -> tryparse(Float64,fields[i+1]),11)
    mc = tryparse(Int,fields[13])
    (any(isnothing,values) || isnothing(mc)) && throw(_unparsable(file,number,fields))
    _TherminatorRow((String(fields[1]),values...,mc))
end

function _canonical_row(file, number, fields)
    length(fields) >= 11 || throw(ArgumentError(
        "$file:$number: expected at least 11 columns (canonical format), found $(length(fields))"))
    id = tryparse(Int,fields[1])
    values = ntuple(i -> tryparse(Float64,fields[i+2]),9)
    (isnothing(id) || any(isnothing,values)) && throw(_unparsable(file,number,fields))
    _CanonicalRow((id,String(fields[2]),values...))
end

# Charm quark and antiquark digits of a PDG id ±(n nr nL nq1 nq2 nq3 nJ).
function _pdg_charm_quarks(id::Integer)
    a = abs(id)
    count(k -> (a ÷ 10^k) % 10 == 4,1:3)
end

"""
    charm_count(row; net=false)

Charm content of a `read_particle_table` row. With `net=false` it is the number
of charm quarks plus antiquarks, the heavy-quark count: 1 for D mesons and
charm baryons and their antiparticles, 2 for charmonia such as J/ψ. With
`net=true` it is the net charm, `+1` for D, `-1` for D̄, `0` for J/ψ.

THERMINATOR rows give `Nc + Nac` or `Nc - Nac`. Canonical rows give the `C`
column for `net=true`; otherwise the count comes from the charm digits of the
PDG id, checked against `C`, since quark-model states without a PDG id (e.g.
`5000440`, an N*) carry numbers whose digits are not quark flavours: `|C|` for
open charm, 2 for a meson (`B = 0`) with `C = 0` whose two quark digits are
both charm, and 0 otherwise.
"""
charm_count(row::_TherminatorRow; net::Bool=false) = net ? row.Nc - row.Nac : row.Nc + row.Nac

function charm_count(row::_CanonicalRow; net::Bool=false)
    net && return row.C
    iszero(row.C) || return abs(row.C)
    hidden = iszero(row.B) && _pdg_charm_quarks(row.id) == 2 && (abs(row.id) ÷ 1000) % 10 == 0
    hidden ? 2.0 : 0.0
end

_degeneracy(row::_CanonicalRow) = row.degeneracy
_degeneracy(row::_TherminatorRow) = 2*row.spin + 1

_fermion(row::_CanonicalRow) = !iszero(row.B)
_fermion(row::_TherminatorRow) = !isinteger(row.spin)

_self_conjugate(row::_TherminatorRow) = row.Nq == row.Naq && row.Ns == row.Nas && row.Nc == row.Nac && iszero(row.I3)

"""
    read_particle_list(file; net=false, charges=row -> (charm_count(row; net),),
                       condition=<any charge ≠ 0>, minmass=0, maxmass=Inf,
                       statistics=:boltzmann, antiparticles=nothing, format=:auto)

`HadronSpecies` from a FastReso particle list in either format of
`read_particle_table`, in the spirit of Fluidum's `readresonancelist`.
`charges(row)` gives the charges of each species; the default is the
`charm_count` of the row, the heavy-quark count `c + c̄` for `net=false` (D and
D̄ → 1, J/ψ → 2) or the net charm for `net=true`. Rows are kept if
`minmass < mass < maxmass` and `condition(row)` holds; by default rows whose
charges all vanish are dropped, so reading a full list yields only the charm
hadrons (with `net=true`, only open charm).

Canonical lists give the degeneracy and list antiparticles as rows of their
own. THERMINATOR lists give the spin, so each row has degeneracy
`2·spin + 1`, and omit the antiparticles: with `antiparticles=true`, the
default for this format, the degeneracy is doubled for every row that is not
its own conjugate (`Nq = Naq`, `Ns = Nas`, `Nc = Nac`, `I3 = 0`, e.g. J/ψ), so
the conjugate shares the charges of the row. That is right for `c + c̄`; for
charges that change sign, such as the net charm, pass `antiparticles=false`
and add the conjugates yourself. `antiparticles=true` is rejected for a
canonical list, which already contains them.

`statistics` is `:boltzmann`, `:boson`, `:fermion`, or `:quantum`, which picks
Fermi-Dirac for baryons (half-integer spin) and Bose-Einstein otherwise.
Masses are in GeV, so the list fits an EOS whose energy unit is GeV.
"""
function read_particle_list(file::AbstractString; net::Bool=false,
                            charges=row -> (charm_count(row; net),),
                            condition=row -> any(!iszero,charges(row)),
                            minmass::Real=0, maxmass::Real=Inf,
                            statistics::Symbol=:boltzmann,
                            antiparticles::Union{Nothing,Bool}=nothing, format::Symbol=:auto)
    statistics in (:boltzmann,:boson,:fermion,:quantum) ||
        throw(ArgumentError("unknown statistics: $statistics"))
    table = read_particle_table(file; format)
    canonical = eltype(table) === _CanonicalRow
    canonical && antiparticles === true && throw(ArgumentError(
        "$file lists antiparticles explicitly; antiparticles=true would count them twice"))
    double = something(antiparticles,!canonical)
    rows = filter(row -> minmass < row.mass < maxmass && condition(row),table)
    isempty(rows) && throw(ArgumentError("no particle of $file passes the selection"))
    N = length(charges(first(rows)))
    map(rows) do row
        q = charges(row)
        length(q) == N || throw(DimensionMismatch("charges must return $N values for every row"))
        g = _degeneracy(row)*(double && !_self_conjugate(row) ? 2 : 1)
        stat = statistics !== :quantum ? statistics : _fermion(row) ? :fermion : :boson
        HadronSpecies(row.mass,g,SVector{N,Float64}(Tuple(q)); statistics=stat)
    end
end
