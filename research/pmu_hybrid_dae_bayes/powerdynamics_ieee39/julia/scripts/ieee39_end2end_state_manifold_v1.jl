# Export the 120-frame native-state sensitivities underlying the frozen PMU
# dictionary.  The trusted analytic exporter is rewritten only in memory to
# select op_m085, extend NF, and persist its already-computed native states.
using PowerDynamics, CSV, DataFrames, Random

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const BASE = joinpath(@__DIR__, "pd39_analytic_second_order_v1.jl")
const DATA = joinpath(ROOT, "output", "t120_multi_op_independent_validation_v1", "op_data", "op_m085")
const OUT = joinpath(ROOT, "output", "ieee39_end2end_single_v1", "state_manifold", "analytic_raw")
const SRC = joinpath(OUT, "ieee39_part1.jl")
mkpath(OUT)

modeltxt = read(joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl"), String)
modeltxt = replace(modeltxt,
    "DATA_DIR = joinpath(pkgdir(PowerDynamics), \"docs\", \"examples\", \"ieee39data\")" =>
    "DATA_DIR = raw\"$DATA\"")
write(SRC, modeltxt)

txt = read(BASE, String)
txt = replace(txt,
    "const OUT = joinpath(ROOT, \"output\", \"analytic_second_order_dae_v1\")" =>
    "const OUT = raw\"$OUT\"")
txt = replace(txt,
    "const SRC = joinpath(pkgdir(PowerDynamics), \"docs\", \"examples\", \"ieee39_part1.jl\")" =>
    "const SRC = raw\"$SRC\"")
txt = replace(txt, "const NF = 30" => "const NF = 120")

# Allocate state-level Taylor coefficients in the exact native order.
needle = "qself = zeros(Float64,NF*nout,length(BUSES)); qcross = zeros(Float64,NF*nout,npair); danalytic = zeros(Float64,NF*nout,length(BUSES))"
inject = needle * "\n" *
    "state_D=zeros(Float64,NF*n,length(BUSES)); state_Q=zeros(Float64,NF*n,length(BUSES)); state_Qcross=zeros(Float64,NF*n,npair)"
occursin(needle, txt) || error("state allocation injection point missing")
txt = replace(txt, needle => inject)

# First derivatives: row zero is the callback/pre-event baseline, matching
# the PMU dictionary; subsequent rows use the existing analytic state.
needle = "    t = (k-1)*DT; xf = first_state(t)"
inject = needle * "\n" *
    "    if k > 1\n        state_D[(k-1)*n+1:k*n,:] .= xf\n    end"
occursin(needle, txt) || error("first-state injection point missing")
txt = replace(txt, needle => inject)

needle = "            qself[(k-1)*nout+1:k*nout,b] .= 0.5 .* (Cfull*full_state(x2self[:,b],x2aself[:,b]) + Hobse[:,b,b])"
inject = "            state_Q[(k-1)*n+1:k*n,b] .= 0.5 .* full_state(x2self[:,b],x2aself[:,b])\n" * needle
occursin(needle, txt) || error("self-state injection point missing")
txt = replace(txt, needle => inject)

needle = "            qcross[(k-1)*nout+1:k*nout,r] .= Cfull*full_state(x2[:,r],x2a[:,r]) + Hobse[:,ii,jj]"
inject = "            state_Qcross[(k-1)*n+1:k*n,r] .= full_state(x2[:,r],x2a[:,r])\n" * needle
occursin(needle, txt) || error("cross-state injection point missing")
txt = replace(txt, needle => inject)

# Binary column-major artifacts are compact and lossless.  Python reshapes
# them with order='F' using the emitted dimensions.
needle = "println(\"analytic_second_order_dae_export_done n=\",n,\" nd=\",nd,\" na=\",na,\" pairs=\",npair)"
inject = "open(joinpath(RES,\"state_D_f64.bin\"),\"w\") do io; write(io,state_D); end\n" *
    "open(joinpath(RES,\"state_Q_f64.bin\"),\"w\") do io; write(io,state_Q); end\n" *
    "open(joinpath(RES,\"state_Qcross_f64.bin\"),\"w\") do io; write(io,state_Qcross); end\n" *
    "CSV.write(joinpath(RES,\"operating_point_state.csv\"),DataFrame(index=1:n,x0=x0))\n" *
    "CSV.write(joinpath(RES,\"state_array_dimensions.csv\"),DataFrame(name=[\"D\",\"Q\",\"Qcross\"],rows=[NF*n,NF*n,NF*n],cols=[length(BUSES),length(BUSES),npair],layout=[\"Float64 column-major\",\"Float64 column-major\",\"Float64 column-major\"]))\n" * needle
occursin(needle, txt) || error("binary export injection point missing")
txt = replace(txt, needle => inject)

mod = Module(Symbol("E2E_STATE_", rand(UInt)))
Base.include_string(mod, txt, "ieee39_end2end_state_manifold_generated.jl")
println("ieee39_end2end_state_manifold_done")
