# IEEE39-END2END-SINGLE-V1 nonlinear truth producer.
#
# This file deliberately exports truth only.  The Python inference stage is
# launched separately from a serialized 32-channel PMU artifact and never
# receives the objects created here.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, Random, SHA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "ieee39_end2end_single_v1", "truth")
const DATA = joinpath(ROOT, "output", "t120_multi_op_independent_validation_v1", "op_data", "op_m085")
const SRC0 = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")
const TAU = 2.0
const DT = 1 / 30
const TEND = 6.0
const AMP = 0.0033
mkpath(OUT)

function build()
    txt = replace(read(SRC0, String),
        r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)" =>
        "DATA_DIR = raw\"$DATA\"")
    mod = Module(Symbol("E2E_SINGLE_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase,
                     NetworkDynamics, OrdinaryDiffEqRosenbrock,
                     OrdinaryDiffEqNonlinearSolve, CSV, DataFrames))
    redirect_stdout(devnull) do
        Base.include_string(mod, txt, "ieee39_end2end_single_v1_model.jl")
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

function install_bus7_callback!(nw, amplitude)
    initialized = Ref(false); p0 = Ref(0.0); q0 = Ref(0.0)
    params = [Symbol("ZIPLoad₊Pset"), Symbol("ZIPLoad₊Qset")]
    affect = ComponentAffect([], params) do u, p, ctx
        if !initialized[]
            p0[] = Float64(p[params[1]]); q0[] = Float64(p[params[2]])
            initialized[] = true
        end
        p[params[1]] = p0[] * (1 + amplitude)
        p[params[2]] = q0[] * (1 + amplitude)
    end
    set_callback!(nw[VIndex(7)], PresetTimeComponentCallback([TAU], affect))
end

function export_solution(label, amplitude)
    nw, s0 = build()
    install_bus7_callback!(nw, amplitude)
    t0 = time()
    sol = solve(ODEProblem(nw, deepcopy(s0), (0.0, TEND)), Rodas5P();
                abstol=1e-11, reltol=1e-11, dtmax=1/60,
                maxiters=10^7, saveat=DT)
    string(sol.retcode) == "Success" || error("$label failed: $(sol.retcode)")
    n = length(uflat(s0)); nt = length(sol.t)
    umat = Matrix{Float64}(undef, nt, n)
    vrows = DataFrame(scenario=String[], time_s=Float64[], sample_index=Int[],
                      bus=Int[], V_re=Float64[], V_im=Float64[])
    for k in 1:nt
        umat[k, :] .= Float64.(sol.u[k])
        for bus in 1:39
            z = complex(Float64(sol(sol.t[k]; idxs=VIndex(bus, :busbar₊u_r))),
                        Float64(sol(sol.t[k]; idxs=VIndex(bus, :busbar₊u_i))))
            push!(vrows, (label, Float64(sol.t[k]), k, bus, real(z), imag(z)))
        end
    end
    state = DataFrame(umat, Symbol.("u" .* string.(1:n)))
    insertcols!(state, 1, :sample_index => collect(1:nt))
    insertcols!(state, 1, :time_s => Float64.(sol.t))
    insertcols!(state, 1, :amplitude => fill(amplitude, nt))
    insertcols!(state, 1, :scenario => fill(label, nt))
    sp = joinpath(OUT, "truth_full_state_" * lowercase(label) * ".csv.gz")
    vp = joinpath(OUT, "truth_full_bus_outputs_" * lowercase(label) * ".csv.gz")
    CSV.write(sp, state; compress=true); CSV.write(vp, vrows; compress=true)
    return (label=label, amplitude=amplitude, runtime_s=time()-t0,
            retcode=string(sol.retcode), n_time=nt, n_state=n,
            state_path=sp, bus_path=vp,
            state_sha256=bytes2hex(sha256(read(sp))), bus_sha256=bytes2hex(sha256(read(vp))))
end

rows = [export_solution("H0", 0.0), export_solution("BUS7_EVENT", AMP)]
CSV.write(joinpath(OUT, "truth_execution.csv"), DataFrame(rows))
println("ieee39_end2end_truth_done rows=", length(rows))
