# Independent operating-point validation bank for LIKELIHOOD-120-CONTRACT-V2.
# The data mutation is limited to the already validated M6 bus-3 P/Q
# operating-point mechanism; event callbacks remain true time-local and do not
# reinitialize stored states. All outputs are excluded from future V3.
using PowerDynamics, CSV, DataFrames, SHA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTROOT = joinpath(ROOT, "output", "t120_multi_op_independent_validation_v1")
const TAG = get(ENV, "OP_TAG", "op_m035")
const M = parse(Float64, get(ENV, "OP_M", "0.35"))
const PKG_DATA = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39data")
const OPDATA = joinpath(OUTROOT, "op_data", TAG)
mkpath(OPDATA)

for f in filter(endswith(".csv"), readdir(PKG_DATA))
    CSV.write(joinpath(OPDATA, f), CSV.read(joinpath(PKG_DATA, f), DataFrame))
end
bus = CSV.read(joinpath(OPDATA, "bus.csv"), DataFrame)
k = findfirst(==(3), Int.(bus.bus))
bus[k, :P] = Float64(bus[k, :P]) * (1 + 0.03*M)
bus[k, :Q] = Float64(bus[k, :Q]) * (1 + 0.03*M)
CSV.write(joinpath(OPDATA, "bus.csv"), bus)

const CANDS = get(ENV, "OP_CANDIDATES", "3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28")
const PAIRS = get(ENV, "OP_PAIRS", "3-4;3-7;3-12;7-12;7-20;8-28;12-20;12-24;16-18;26-28")
const MAGPAIRS = get(ENV, "OP_MAG_PAIRS", "0.0001,0.0002;0.0005,0.001;0.002,0.004;0.01,0.025")
const OPMODE = get(ENV, "OP_MODE", "DEV")
ENV["L120_MODE"] = OPMODE
ENV["L120_OUT"] = "t120_multi_op_independent_validation_v1/physical/$(TAG)"
ENV["L120_MAG_PAIRS"] = MAGPAIRS
ENV["L120_HORIZON"] = "6.1"
ENV["L120_REUSE"] = "1"
ENV["L120_DATASRC"] = OPDATA
ENV["MULTI_CANDIDATES"] = CANDS
ENV["MULTI_PAIRS"] = PAIRS
if OPMODE == "SINGLE"
    ENV["L120_AMPS"] = get(ENV, "OP_AMPS", "-0.005,0.0,0.005")
end

include(joinpath(@__DIR__, "pd39_likelihood_120_physical_v2.jl"))

files = sort(readdir(OPDATA)); digest = bytes2hex(sha256(reduce(vcat, [read(joinpath(OPDATA, f)) for f in files])))
mfpath = joinpath(OUTROOT, "operating_points.csv")
row = DataFrame(op_tag=[TAG], op_m=[M], mutation=["bus3.P,Q *= 1+0.03*m"], data_sha256=[digest], candidates=[CANDS], pairs=[PAIRS], magnitude_pairs=[MAGPAIRS], horizon_s=[6.1])
if isfile(mfpath)
    old = CSV.read(mfpath, DataFrame); CSV.write(mfpath, vcat(old, row))
else
    CSV.write(mfpath, row)
end
println("t120_multiop_validation_done tag=", TAG, " m=", M)
