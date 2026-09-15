# Minimal production-event flow maps for EVENT-MAP-ALIGNMENT-PILOT-V1.
# This script is audit-only: it uses the frozen nominal plant and the exact
# callback/save contract, without changing the simulator or estimator.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random, Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "event_map_alignment_pilot_v1")
const FLOW = joinpath(OUT, "flow_maps")
mkpath(FLOW)
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")
const TAU = 2.0
const HORIZON = 5.0
const SAVEAT = 1 / 30
const TOL = 1e-11
const DTMAX = 1 / 60
const FORCE = true

function build()
    mod = Module(Symbol("EVENT_MAP_FLOW_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase,
        NetworkDynamics, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve))
    redirect_stdout(devnull) do
        Base.include_string(mod, read(SRC, String), "event_map_flow_ieee39.jl")
    end
    nw = getfield(mod, :nw)
    Core.eval(mod, quote
        formula = @initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
        set_initformula!($nw[VIndex(31)], formula)
        set_initformula!($nw[VIndex(39)], formula)
    end)
    pfs = Base.invokelatest(solve_powerflow, nw; verbose=false)
    s0 = Base.invokelatest(initialize_from_pf!, nw; pfs=pfs, verbose=false)
    return nw, s0
end

function add_event!(nw, bus, amp)
    syms = [Symbol("ZIPLoad₊Pset"), Symbol("ZIPLoad₊Qset")]
    aff = ComponentAffect([], syms) do u, p, ctx
        p[syms[1]] *= (1 + amp)
        p[syms[2]] *= (1 + amp)
    end
    set_callback!(nw[VIndex(bus)], PresetTimeComponentCallback([TAU], aff))
end

function run_case(label, events)
    path = joinpath(FLOW, label * ".csv")
    if isfile(path) && !FORCE
        d = CSV.read(path, DataFrame)
        return DataFrame(case=[label], events=[join(string.(events), ";")], retcode=["RESUMED"],
                         n_save=[nrow(d)], finite=[all(isfinite, Matrix(d[:, 5:end]))], path=[path])
    end
    nw, s0 = build()
    for (bus, amp) in events
        add_event!(nw, bus, amp)
    end
    sol = solve(ODEProblem(nw, s0, (0.0, HORIZON)), Rodas5P(); abstol=TOL, reltol=TOL,
                 dtmax=DTMAX, maxiters=10^7, saveat=SAVEAT)
    ts = Float64.(sol.t)
    mat = reduce(vcat, [reshape(Float64.(sol(t)), 1, :) for t in ts])
    names = [Symbol("u", i) for i in 1:size(mat, 2)]
    df = DataFrame(mat, names)
    insertcols!(df, 1, :time => ts)
    insertcols!(df, 1, :case => fill(label, nrow(df)))
    CSV.write(path, df)
    DataFrame(case=[label], events=[join(string.(events), ";")], retcode=[string(sol.retcode)],
              n_save=[length(ts)], finite=[all(isfinite, mat)], path=[path])
end

rows = DataFrame(case=String[], events=String[], retcode=String[], n_save=Int[], finite=Bool[], path=String[])
for h in (0.0025, 0.005, 0.01)
    token = replace(string(h), "." => "p")
    cases = [("base_h" * token, Tuple{Int,Float64}[]),
             ("self7_p_h" * token, [(7, h)]),
             ("self7_m_h" * token, [(7, -h)]),
             ("cross712_pp_h" * token, [(7, h), (12, h)]),
             ("cross712_pm_h" * token, [(7, h), (12, -h)]),
             ("cross712_mp_h" * token, [(7, -h), (12, h)]),
             ("cross712_mm_h" * token, [(7, -h), (12, -h)])]
    for (label, ev) in cases
        println("running ", label)
        append!(rows, run_case(label, ev))
    end
end
CSV.write(joinpath(OUT, "flow_manifest.csv"), rows)
println("event_map_alignment_flow_done cases=", nrow(rows), " output=", OUT)
