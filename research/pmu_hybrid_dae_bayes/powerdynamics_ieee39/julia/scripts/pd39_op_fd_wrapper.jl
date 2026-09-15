# Fresh nonlinear-TDS central-difference bank for the operating-point
# second-order robustness audit.  The bank is derivative-validation only and
# is excluded from any future prospective V3 split.
using PowerDynamics, CSV, DataFrames, Random, SHA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTROOT = joinpath(ROOT, "output", "second_order_op_robustness_v1")
const TAG = get(ENV, "OP_TAG", "nominal")
const M = parse(Float64, get(ENV, "OP_M", "0.0"))
const ATLAS_SCRIPT = joinpath(@__DIR__, "pd_load_response_atlas.jl")
const PKG_DATA = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39data")
const OPDATA = joinpath(OUTROOT, "fd_data", TAG)
const ATLAS_OUT = "fd_" * TAG
mkpath(OPDATA)

function copy_and_mutate()
    for f in filter(endswith(".csv"), readdir(PKG_DATA))
        CSV.write(joinpath(OPDATA, f), CSV.read(joinpath(PKG_DATA, f), DataFrame))
    end
    bus = CSV.read(joinpath(OPDATA, "bus.csv"), DataFrame)
    k = findfirst(==(3), Int.(bus.bus))
    bus[k, :P] = Float64(bus[k, :P]) * (1 + 0.03*M)
    bus[k, :Q] = Float64(bus[k, :Q]) * (1 + 0.03*M)
    CSV.write(joinpath(OPDATA, "bus.csv"), bus)
end

copy_and_mutate()

txt = read(ATLAS_SCRIPT, String)
txt = replace(txt,
    "const OUT=joinpath(ROOT,\"output\",get(ENV,\"ATLAS_OUT\",\"load_response_atlas_v1\"))" =>
    "const OUT=raw\"" * joinpath(OUTROOT, ATLAS_OUT) * "\"")
txt = replace(txt,
    "const DATASRC=joinpath(pkgdir(PowerDynamics),\"docs\",\"examples\",\"ieee39data\")" =>
    "const DATASRC=raw\"" * OPDATA * "\"")
txt = replace(txt, "const EPS=[0.0025,-0.0025,0.005,-0.005,0.01,-0.01,0.02,-0.02,0.10]" => "const EPS=[0.0025,-0.0025,0.005,-0.005,0.01,-0.01]")

ENV["ATLAS_OUT"] = ATLAS_OUT
ENV["ATLAS_CANDIDATES"] = "3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28"
ENV["ATLAS_AMPS"] = "0.0025,-0.0025,0.005,-0.005,0.01,-0.01"
ENV["ATLAS_REPS"] = "1"
mod = Module(Symbol("OPFD_", rand(UInt)))
Base.include_string(mod, txt, "pd_load_response_atlas_$(TAG).jl")
println("operating_point_fd_done tag=", TAG, " m=", M)
