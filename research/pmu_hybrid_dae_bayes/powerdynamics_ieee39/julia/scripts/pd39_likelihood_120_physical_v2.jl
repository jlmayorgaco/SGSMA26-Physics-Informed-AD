# Dedicated T=120 physical bank for LIKELIHOOD-120-CONTRACT-V2.
# This wrapper reuses the frozen native multi-event harness, changing only
# the output namespace and integration horizon.  It preserves the production
# callback semantics and is excluded from prospective V3.
using PowerDynamics, CSV, DataFrames, Random

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const BASE = joinpath(@__DIR__, "pd_load_multi_atlas.jl")
const MODE = get(ENV, "L120_MODE", "QIJ")
const OUTNAME = get(ENV, "L120_OUT", "likelihood_120_contract_v2/physical_dictionary")
const HORIZON = get(ENV, "L120_HORIZON", "6.1")

ENV["MULTI_OUT"] = OUTNAME
ENV["MULTI_MODE"] = MODE
ENV["MULTI_REUSE"] = get(ENV, "L120_REUSE", MODE == "QIJ" ? "1" : "0")
if MODE == "SINGLE"
    ENV["MULTI_AMPS"] = get(ENV, "L120_AMPS", "-0.005,0.0,0.005")
else
    ENV["MULTI_H"] = get(ENV, "L120_H", "0.005")
end

txt = read(BASE, String)
# `include_string` gives the included source a synthetic filename, so its
# `@__DIR__` is not the native script directory.  Pin the base harness root
# explicitly to keep all generated artifacts inside this repository.
const ROOT_LIT = replace(ROOT, "\\" => "/")
txt = replace(txt, r"const ROOT=normpath\(joinpath\(@__DIR__,\"\.\.\",\"\.\.\"\)\)" => "const ROOT=raw\"$ROOT_LIT\"")
txt = replace(txt, "(0.0,5.0)" => "(0.0," * HORIZON * ")")
mod = Module(Symbol("L120_PHYSICAL_", rand(UInt)))
Base.include_string(mod, txt, "pd_load_multi_atlas_l120.jl")
println("resolved_out=", getfield(mod, :OUT), " resolved_res=", getfield(mod, :RES))
println("likelihood_120_physical_done mode=", MODE, " horizon=", HORIZON,
        " output=", joinpath(ROOT, "output", OUTNAME))
