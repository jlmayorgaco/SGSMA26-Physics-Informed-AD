# HARD-PAIR-CROSS-SECOND-ORDER-REPLICATION-V1 production TDS executor.
# The manifest is frozen before execution and contains only 26-28, 3-18 and
# 16-18 at the three already excluded operating points.  Every solve exports
# full state and production voltage samples from the same numerical solution.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, Random, SHA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "hard_pair_cross_second_order_replication_v1")
const RES = joinpath(OUT, "results"); const PHYS = joinpath(OUT, "physical")
const MANIFEST = joinpath(RES, "matched_stencil_manifest.csv")
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")
const TAU = 2.0; const DT = 1 / 30; const KMAX = 45; const TEND = TAU + KMAX * DT
mkpath(RES); mkpath(PHYS)

function build(data)
    txt = replace(read(SRC, String),
        r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)" => "DATA_DIR = raw\"$data\"")
    mod = Module(Symbol("HARD_CROSS_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase,
                     NetworkDynamics, OrdinaryDiffEqRosenbrock,
                     OrdinaryDiffEqNonlinearSolve, CSV, DataFrames))
    redirect_stdout(devnull) do
        Base.include_string(mod, txt, "hard_cross_ieee39.jl")
    end
    nw = getfield(mod, :nw)
    Core.eval(mod, quote
        formula = @initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
        set_initformula!($nw[VIndex(31)], formula)
        set_initformula!($nw[VIndex(39)], formula)
    end)
    pf = Base.invokelatest(solve_powerflow, nw; verbose=false)
    s0 = Base.invokelatest(initialize_from_pf!, nw; pfs=pf, verbose=false)
    return nw, s0
end

function callbacks!(nw)
    refs = Dict{Int,Ref{Float64}}()
    for bus in (3, 16, 18, 26, 28)
        ref = Ref(0.0); refs[bus] = ref
        initialized = Ref(false); p0 = Ref(0.0); q0 = Ref(0.0)
        params = [Symbol("ZIPLoad₊Pset"), Symbol("ZIPLoad₊Qset")]
        affect = ComponentAffect([], params) do u, p, ctx
            if !initialized[]
                p0[] = Float64(p[params[1]]); q0[] = Float64(p[params[2]])
                initialized[] = true
            end
            p[params[1]] = p0[] * (1 + ref[])
            p[params[2]] = q0[] * (1 + ref[])
        end
        set_callback!(nw[VIndex(bus)], PresetTimeComponentCallback([TAU], affect))
    end
    refs
end

function export_run(nw, s0, refs, row)
    op = String(row.op_tag); pid = String(row.point_id)
    sdir = joinpath(PHYS, op, "state"); vdir = joinpath(PHYS, op, "voltage")
    mkpath(sdir); mkpath(vdir)
    spath = joinpath(sdir, pid * ".csv"); vpath = joinpath(vdir, pid * ".csv")
    if isfile(spath) && isfile(vpath)
        return (true, 0.0, "CHECKPOINT", spath, vpath)
    end
    for ref in values(refs); ref[] = 0.0; end
    bi = Int(row.bus_i); bj = Int(row.bus_j)
    refs[bi][] = Float64(row.amplitude_i); refs[bj][] = Float64(row.amplitude_j)
    t0 = time()
    try
        sol = solve(ODEProblem(nw, deepcopy(s0), (0.0, TEND)), Rodas5P();
                    abstol=1e-11, reltol=1e-11, dtmax=1/60,
                    maxiters=10^7, saveat=DT)
        ret = string(sol.retcode)
        ret == "Success" || return (false, time()-t0, ret, spath, vpath)
        n = length(uflat(s0)); keep = findall(t -> t > TAU + 1e-10 && t <= TEND + 1e-10, Float64.(sol.t))
        length(keep) == KMAX || error("expected $KMAX post-event samples, got $(length(keep))")
        umat = Matrix{Float64}(undef, KMAX, n); times = Vector{Float64}(undef, KMAX)
        volts = DataFrame(time_s=Float64[], sample_index=Int[], bus=Int[], V_re=Float64[], V_im=Float64[])
        for (k, ix) in enumerate(keep)
            t = Float64(sol.t[ix]); times[k] = t; umat[k, :] .= Float64.(sol.u[ix])
            for bus in 1:39
                z = complex(Float64(sol(t; idxs=VIndex(bus, :busbar₊u_r))),
                            Float64(sol(t; idxs=VIndex(bus, :busbar₊u_i))))
                push!(volts, (t, k, bus, real(z), imag(z)))
            end
        end
        state = DataFrame(umat, Symbol.("u" .* string.(1:n)))
        insertcols!(state, 1, :sample_index => collect(1:KMAX)); insertcols!(state, 1, :time_s => times)
        insertcols!(state, 1, :amplitude_j => fill(Float64(row.amplitude_j), KMAX))
        insertcols!(state, 1, :amplitude_i => fill(Float64(row.amplitude_i), KMAX))
        insertcols!(state, 1, :point_id => fill(pid, KMAX))
        CSV.write(spath, state); CSV.write(vpath, volts)
        return (true, time()-t0, ret, spath, vpath)
    catch err
        return (false, time()-t0, sprint(showerror, err), spath, vpath)
    end
end

function main()
    isfile(MANIFEST) || error("preregistered manifest missing")
    mf = CSV.read(MANIFEST, DataFrame); outpath = joinpath(RES, "tds_execution_manifest.csv")
    out = isfile(outpath) ? CSV.read(outpath, DataFrame) : DataFrame(
        point_id=String[], op_tag=String[], pair=String[], status=String[], runtime_s=Float64[], retcode=String[],
        state_path=String[], voltage_path=String[], state_sha256=String[], voltage_sha256=String[])
    for group in groupby(mf, :op_tag)
        op = String(group.op_tag[1]); data = joinpath(ROOT, "output", "t120_multi_op_independent_validation_v1", "op_data", op)
        nw, s0 = build(data); refs = callbacks!(nw)
        for row in eachrow(group)
            pid = String(row.point_id)
            if pid in out.point_id; println(op, " ", pid, " CHECKPOINT"); continue; end
            ok, runtime, ret, spath, vpath = export_run(nw, s0, refs, row)
            sh = isfile(spath) ? bytes2hex(sha256(read(spath))) : ""; vh = isfile(vpath) ? bytes2hex(sha256(read(vpath))) : ""
            push!(out, (pid, op, String(row.pair), ok ? "EXECUTED_SUCCESS" : "EXECUTED_FAIL", runtime, ret, spath, vpath, sh, vh))
            CSV.write(outpath, out); println(op, " ", row.pair, " ", pid, " ", ok, " ", round(runtime, digits=2)); flush(stdout); GC.gc()
        end
    end
    CSV.write(outpath, out)
end

main()

