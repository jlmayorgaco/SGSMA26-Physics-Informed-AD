# Operating-point derivative-validation wrapper for ANALYTIC-SECOND-ORDER-OP-ROBUSTNESS-V1.
# It runs the existing frozen analytic exporter against one contract-valid M6
# operating point.  The exporter is evaluated in an isolated module and its
# source/data paths are rewritten only in memory.
using PowerDynamics, CSV, DataFrames, Random, LinearAlgebra, SHA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const JULIA_ROOT = dirname(@__DIR__)
const OUTROOT = joinpath(ROOT, "output", "second_order_op_robustness_v1")
const TAG = get(ENV, "OP_TAG", "nominal")
const M = parse(Float64, get(ENV, "OP_M", "0.0"))
const EXP_SCRIPT = joinpath(@__DIR__, "pd39_analytic_second_order_v1.jl")
const PKG_DATA = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39data")
const OPDATA = joinpath(OUTROOT, "op_data", TAG)
mkpath(OPDATA)

function copy_and_mutate()
    for f in readdir(PKG_DATA)
        src = joinpath(PKG_DATA, f); dst = joinpath(OPDATA, f)
        CSV.write(dst, CSV.read(src, DataFrame))
    end
    bus = CSV.read(joinpath(OPDATA, "bus.csv"), DataFrame)
    # Exact frozen E06 M6 contract: bus 3 P/Q are scaled by 1+0.03*m.
    k = findfirst(==(3), Int.(bus.bus))
    bus[k, :P] = Float64(bus[k, :P]) * (1 + 0.03*M)
    bus[k, :Q] = Float64(bus[k, :Q]) * (1 + 0.03*M)
    CSV.write(joinpath(OPDATA, "bus.csv"), bus)
    meta = DataFrame(op_tag=[TAG], op_m=[M], mutation=["bus3.P,Q *= 1+0.03*m"],
                     data_sha256=[bytes2hex(sha256(reduce(vcat, [read(joinpath(OPDATA,f)) for f in sort(readdir(OPDATA))])))])
    CSV.write(joinpath(OUTROOT, "operating_point_mutation_"*TAG*".csv"), meta)
end

copy_and_mutate()

# Rewrite the existing exporter in memory so the frozen implementation is
# reused byte-for-byte, while OUT/SRC point to this operating point.
txt = read(EXP_SCRIPT, String)
txt = replace(txt,
    "const OUT = joinpath(ROOT, \"output\", \"analytic_second_order_dae_v1\")" =>
    "const OUT = raw\"" * joinpath(OUTROOT, "analytic_"*TAG) * "\"")
txt = replace(txt,
    "const SRC = joinpath(pkgdir(PowerDynamics), \"docs\", \"examples\", \"ieee39_part1.jl\")" =>
    "const SRC = raw\"" * joinpath(OPDATA, "ieee39_part1.jl") * "\"")
# Install a source file with the OP data directory embedded.
src_txt = read(joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl"), String)
src_txt = replace(src_txt, "DATA_DIR = joinpath(pkgdir(PowerDynamics), \"docs\", \"examples\", \"ieee39data\")" => "DATA_DIR = raw\"" * OPDATA * "\"")
write(joinpath(OPDATA, "ieee39_part1.jl"), src_txt)

# Persist OP-level state and algebraic conditioning after the exporter has
# created x0/A_aa/As.  This is injected immediately after the Schur reduction.
needle = "As = A_dd - A_da * (A_aa \\ A_ad)"
inject = needle * "\n" *
    "CSV.write(joinpath(RES, \"operating_point_state.csv\"), DataFrame(index=1:n, x0=x0))\n" *
    "CSV.write(joinpath(RES, \"operating_point_params.csv\"), DataFrame(index=1:np, p0=p0))\n" *
    "sv=svdvals(A_aa); CSV.write(joinpath(RES, \"operating_point_conditioning.csv\"), DataFrame(op_tag=[\"$(TAG)\"], op_m=[$(M)], sigma_min_gz=[minimum(sv)], sigma_max_gz=[maximum(sv)], kappa_gz=[maximum(sv)/minimum(sv)], spectral_abscissa=[maximum(real.(eigvals(As)))], state_norm=[norm(x0)]))"
txt = replace(txt, needle => inject)

mod = Module(Symbol("OPROB_", rand(UInt)))
Base.include_string(mod, txt, "pd39_analytic_second_order_v1_$(TAG).jl")
println("operating_point_export_done tag=", TAG, " m=", M)
