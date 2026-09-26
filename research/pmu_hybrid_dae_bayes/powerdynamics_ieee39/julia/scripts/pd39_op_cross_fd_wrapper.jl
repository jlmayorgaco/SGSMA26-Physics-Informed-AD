# Cross-term central-difference bank for selected, preregistered pairs.
using PowerDynamics, CSV, DataFrames, Random
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTROOT = joinpath(ROOT, "output", "second_order_op_robustness_v1")
const TAG = get(ENV, "OP_TAG", "nominal")
const M = parse(Float64, get(ENV, "OP_M", "0.0"))
const H = parse(Float64, get(ENV, "OP_H", "0.005"))
const MULTI_SCRIPT = joinpath(@__DIR__, "pd_load_multi_atlas.jl")
const PKG_DATA = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39data")
const OPDATA = joinpath(OUTROOT, "fd_data", TAG)
mkpath(OPDATA)
for f in filter(endswith(".csv"), readdir(PKG_DATA))
    CSV.write(joinpath(OPDATA, f), CSV.read(joinpath(PKG_DATA, f), DataFrame))
end
bus = CSV.read(joinpath(OPDATA, "bus.csv"), DataFrame); k=findfirst(==(3),Int.(bus.bus))
bus[k,:P]=Float64(bus[k,:P])*(1+0.03*M); bus[k,:Q]=Float64(bus[k,:Q])*(1+0.03*M); CSV.write(joinpath(OPDATA,"bus.csv"),bus)
txt=read(MULTI_SCRIPT,String)
txt=replace(txt,"const OUT=joinpath(ROOT,\"output\",get(ENV,\"MULTI_OUT\",\"load_multi_bayes_v1\"))"=>"const OUT=raw\""*joinpath(OUTROOT,"cross_"*TAG*"_h"*replace(string(H),"."=>"p"))*"\"")
txt=replace(txt,"const DATASRC=joinpath(pkgdir(PowerDynamics),\"docs\",\"examples\",\"ieee39data\")"=>"const DATASRC=raw\""*OPDATA*"\"")
ENV["MULTI_OUT"]="second_order_op_robustness_v1/cross_"*TAG*"_h"*replace(string(H),"."=>"p")
ENV["MULTI_CANDIDATES"]="3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28"
ENV["MULTI_PAIRS"]="3-4;3-7;3-12;7-12;7-20;8-28;12-15;20-21;23-24;27-28"
ENV["MULTI_REPS"]="1"; ENV["MULTI_MODE"]="QIJ"; ENV["MULTI_H"]=string(H); ENV["MULTI_REUSE"]="1"
mod=Module(Symbol("OPCROSS_",rand(UInt))); Base.include_string(mod,txt,"pd_load_multi_atlas_$(TAG)_$(H).jl")
println("operating_point_cross_fd_done tag=",TAG," h=",H)
