# Minimal, checkpoint-safe nonlinear TDS driver for the targeted margin audit.
# The input point list is frozen before execution.  The model is rebuilt at
# the specified operating point (the validated M6 bus-3 P/Q mutation), and
# the production callback changes ZIP parameters at t=2 without
# reinitialising the stored state.  No estimator/statistical code is present.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, SHA, Random

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "targeted_nonlinear_margin_closure_v1")
const RES = joinpath(OUT, "results")
const PHYS = joinpath(OUT, "physical")
const POINTS = get(ENV, "TARGETED_POINTS_FILE", joinpath(RES, "stage_a_points.csv"))
const HORIZON = parse(Float64, get(ENV, "TARGETED_HORIZON", "6.1"))
mkpath(PHYS); mkpath(joinpath(PHYS, "results")); mkpath(joinpath(PHYS, "checkpoints"))

const PKGEX = joinpath(pkgdir(PowerDynamics), "docs", "examples")
const SRC = joinpath(PKGEX, "ieee39_part1.jl")

function build(data::String)
    txt = replace(read(SRC, String), r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)" => "DATA_DIR = raw\"$data\"")
    mod = Module(Symbol("TARGETED_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase,
                     NetworkDynamics, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve,
                     CSV, DataFrames))
    redirect_stdout(devnull) do; Base.include_string(mod, txt, "targeted_part1.jl"); end
    nw = getfield(mod, :nw)
    Core.eval(mod, quote
        formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
        set_initformula!($nw[VIndex(31)], formula)
        set_initformula!($nw[VIndex(39)], formula)
    end)
    pfs = Base.invokelatest(solve_powerflow, nw; verbose=false)
    s0 = Base.invokelatest(initialize_from_pf!, nw; pfs=pfs, verbose=false)
    return nw, s0
end

function tag(a::Float64)
    replace(string(a), "-"=>"m", "."=>"p")
end

function make_callbacks!(nw, supports::Vector{Int})
    refs = Dict{Int,Ref{Float64}}()
    for bus in supports
        ref = Ref(0.0); refs[bus] = ref
        init = Ref(false); bp = Ref(0.0); bq = Ref(0.0)
        syms = [Symbol("ZIPLoad₊Pset"), Symbol("ZIPLoad₊Qset")]
        aff = ComponentAffect([], syms) do u, p, ctx
            if !init[]
                bp[] = float(p[syms[1]]); bq[] = float(p[syms[2]]); init[] = true
            end
            p[syms[1]] = bp[] * (1 + ref[]); p[syms[2]] = bq[] * (1 + ref[])
        end
        set_callback!(nw[VIndex(bus)], PresetTimeComponentCallback([2.0], aff))
    end
    return refs
end

function run_point(nw_template, s0_template, r, groupdir::String)
    typ = String(r.competitor_type); sp = split(String(r.competitor_support), "-")
    buses = typ == "single" ? [parse(Int, sp[1])] : [parse(Int, sp[1]), parse(Int, sp[2])]
    amps = typ == "single" ? [Float64(r.b1)] : [Float64(r.b1), Float64(r.b2)]
    nw = deepcopy(nw_template); s0 = deepcopy(s0_template); refs = make_callbacks!(nw, buses)
    for (b,a) in zip(buses, amps); refs[b][] = a; end
    tid = String(r.trajectory_id); path = joinpath(groupdir, tid * ".csv")
    if isfile(path)
        return (tid, true, 0.0, path, "CHECKPOINT")
    end
    t0 = time()
    ok = true; ret = "ERROR"
    try
        sol = solve(ODEProblem(nw, s0, (0.0, HORIZON)), Rodas5P(); abstol=1e-9, reltol=1e-9,
                    saveat=1/30, dtmax=1/60, maxiters=10^7)
        ret = string(sol.retcode); ok = ret == "Success" && all(isfinite, sol.u[end])
        rows = DataFrame(time=Float64[], bus=Int[], V_re=Float64[], V_im=Float64[])
        for t in sol.t, b in 1:39
            z = complex(float(sol(t; idxs=VIndex(b, :busbar₊u_r))), float(sol(t; idxs=VIndex(b, :busbar₊u_i))))
            push!(rows, (t, b, real(z), imag(z)))
        end
        ok && CSV.write(path, rows)
    catch err
        ret = sprint(showerror, err); ok = false
    end
    dt = time() - t0
    if ok
        open(joinpath(groupdir, tid * ".done"), "w") do io
            write(io, "retcode=$(ret),frames=", "$(round(Int, HORIZON*30+1)),seconds=$dt\n")
        end
    end
    return (tid, ok, dt, path, ret)
end

function main()
    pts = CSV.read(POINTS, DataFrame)
    isempty(pts) && error("No target points in $POINTS")
    # Point rows are already grouped by OP; no point is silently changed or
    # inferred from the TDS output.
    manifest_path = joinpath(RES, "tds_execution_manifest.csv")
    outrows = isfile(manifest_path) ? CSV.read(manifest_path, DataFrame) : DataFrame(trajectory_id=String[], op_tag=String[], op_m=Float64[], competitor_type=String[], competitor_support=String[], b1=Float64[], b2=Float64[], status=String[], path=String[], runtime_s=Float64[], retcode=String[], sha256=String[])
    for (op, m) in (("op_m035",0.35),("op_m085",0.85),("op_m125",1.25))
        p = pts[pts.op_tag .== op, :]; nrow(p)==0 && continue
        data = joinpath(ROOT, "output", "t120_multi_op_independent_validation_v1", "op_data", op)
        isdir(data) || error("missing frozen operating-point data: $data")
        nw, s0 = build(data)
        groupdir = joinpath(PHYS, op, "results"); mkpath(groupdir)
        for r in eachrow(p)
            x = run_point(nw, s0, r, groupdir)
            h = isfile(x[4]) ? bytes2hex(sha256(read(x[4]))) : ""
            if !(x[1] in outrows.trajectory_id)
                push!(outrows, (x[1],op,m,String(r.competitor_type),String(r.competitor_support),Float64(r.b1),Float64(r.b2),x[2] ? x[5] == "CHECKPOINT" ? "CHECKPOINT" : "EXECUTED_SUCCESS" : "EXECUTED_FAIL",x[4],x[3],x[5],h))
            end
            CSV.write(manifest_path, outrows)
            println(op, " ", x[1], " ", x[2], " ", x[3], " ", x[5]); flush(stdout)
        end
        GC.gc()
    end
    CSV.write(manifest_path, outrows)
end

main()
