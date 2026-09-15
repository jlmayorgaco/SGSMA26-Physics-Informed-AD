# Representative TDS flow/tolerance/onset probe for SECOND-ORDER-DISCREPANCY-
# CLOSURE-V1.  This is intentionally small: Bus 7 self and (7,12) cross,
# with no V3/T120 data.  Every trajectory starts from the nominal initialized
# state and changes only the ZIP parameter at t=2 s.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random, Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "second_order_discrepancy_closure_v1", "flow_probe")
mkpath(OUT)
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")

function build()
    mod = Module(Symbol("DISC_FLOW_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase,
        NetworkDynamics, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve))
    redirect_stdout(devnull) do
        Base.include_string(mod, read(SRC, String), "flow_probe_ieee39.jl")
    end
    nw = getfield(mod, :nw)
    Core.eval(mod, quote
        formula = @initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
        set_initformula!($nw[VIndex(31)], formula)
        set_initformula!($nw[VIndex(39)], formula)
    end)
    pfs = Base.invokelatest(solve_powerflow, nw; verbose=false)
    s0 = Base.invokelatest(initialize_from_pf!, nw; pfs=pfs, verbose=false)
    nw, s0
end

function run_case(label, events, tol, dtmax, saveat, horizon)
    nw, s0 = build()
    for (bus, amp) in events
        syms = [Symbol("ZIPLoad₊Pset"), Symbol("ZIPLoad₊Qset")]
        aff = ComponentAffect([], syms) do u, p, ctx
            p[syms[1]] *= (1 + amp); p[syms[2]] *= (1 + amp)
        end
        set_callback!(nw[VIndex(bus)], PresetTimeComponentCallback([2.0], aff))
    end
    sol = solve(ODEProblem(nw, s0, (0.0, horizon)), Rodas5P();
                abstol=tol, reltol=tol, dtmax=dtmax, maxiters=10^7, saveat=saveat)
    ts = Float64.(sol.t)
    mat = reduce(vcat, [reshape(Float64.(sol(t)), 1, :) for t in ts])
    names = [Symbol("u", i) for i in 1:size(mat, 2)]
    df = DataFrame(mat, names)
    insertcols!(df, 1, :time => ts)
    insertcols!(df, 1, :saveat => fill(saveat, nrow(df)))
    insertcols!(df, 1, :dtmax => fill(dtmax, nrow(df)))
    insertcols!(df, 1, :tol => fill(tol, nrow(df)))
    insertcols!(df, 1, :case => fill(label, nrow(df)))
    path = joinpath(OUT, label * ".csv")
    CSV.write(path, df)
    status = DataFrame(case=[label], tol=[tol], dtmax=[dtmax], saveat=[saveat],
                       retcode=[string(sol.retcode)], n_save=[length(ts)],
                       min_time_step=[minimum(diff(ts))], max_time_step=[maximum(diff(ts))],
                       finite=[all(isfinite, mat)])
    status
end

rows = DataFrame(case=String[], tol=Float64[], dtmax=Float64[], saveat=Float64[],
                 retcode=String[], n_save=Int[], min_time_step=Float64[],
                 max_time_step=Float64[], finite=Bool[])
# Tolerance hierarchy uses the same event, amplitudes and output contract.
for (tol, dtmax) in ((1e-7, 1/15), (1e-9, 1/30), (1e-11, 1/60))
    for (kind, events) in (("self7", [(7,0.005)]), ("self7m", [(7,-0.005)]),
                           ("cross712pp", [(7,0.005),(12,0.005)]),
                           ("cross712pm", [(7,0.005),(12,-0.005)]),
                           ("cross712mp", [(7,-0.005),(12,0.005)]),
                           ("cross712mm", [(7,-0.005),(12,-0.005)]))
        label = @sprintf("tol%s_dt%s_%s", replace(string(tol),"."=>"p","-"=>"m"), replace(string(dtmax),"."=>"p"), kind)
        append!(rows, run_case(label, events, tol, dtmax, 1/30, 5.0))
    end
end
# A short-time sequence isolates the local post-event flow (Bus 7 self).
for dt in (1/30, 1/60, 1/120)
    token = replace(string(dt),"."=>"p")
    append!(rows, run_case(@sprintf("short_dt%s_plus", token), [(7,0.005)], 1e-11, dt, dt, 2.0 + 8dt))
    append!(rows, run_case(@sprintf("short_dt%s_minus", token), [(7,-0.005)], 1e-11, dt, dt, 2.0 + 8dt))
end
CSV.write(joinpath(OUT, "probe_manifest.csv"), rows)
println("discrepancy_flow_probe_done cases=$(nrow(rows)) output=$(OUT)")
