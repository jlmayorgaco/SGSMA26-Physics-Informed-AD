using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Serialization, DelimitedFiles

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "weak_weak_resolution_limit_v1", "native")
mkpath(OUT)
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")

const OBS = [2,5,6,10,19,22,29,39]
const EDGE_FOR = [11,37,18,35,31,1,8,5]
const NATIVE_FOR = [false,false,true,true,true,false,false,false]

function build()
    mod = Module(Symbol("WEAKWEAK_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, CSV, DataFrames))
    redirect_stdout(devnull) do
        Base.include_string(mod, read(SRC, String), "weak_weak_ieee39.jl")
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

function channels()
    pout = Any[]
    for b in OBS
        push!(pout, VIndex(b, :busbar₊u_r)); push!(pout, VIndex(b, :busbar₊u_i))
    end
    # The PMU current convention matches e04_export_e04_model and h6.
    for (e,native) in zip(EDGE_FOR, NATIVE_FOR)
        side = native ? :src : :dst
        push!(pout, EIndex(e, Symbol(side, "₊i_r"))); push!(pout, EIndex(e, Symbol(side, "₊i_i")))
    end
    pout
end

ok = true; err = ""
try
    nw, s0 = build()
    desc = linearize_network(s0)
    red = reduce_dae(desc)
    M = Matrix(desc.M); Afull = Matrix(desc.A); Ared = Matrix(red.A)
    syms = SII.variable_symbols(nw); ps = SII.parameter_symbols(nw)
    d = diag(M); didx = findall(!iszero, d)
    pout = channels(); lsys = linearize_network(s0; in=VIndex(1, :busbar₊u_r), out=pout); lred = reduce_dae(lsys); C = Matrix(lred.C)
    CSV.write(joinpath(OUT, "state_order.csv"), DataFrame(index=collect(1:length(syms)), symbol=string.(syms), mass=d, kind=[iszero(x) ? "algebraic" : "differential" for x in d]))
    CSV.write(joinpath(OUT, "reduced_state_order.csv"), DataFrame(index=collect(1:length(didx)), full_index=didx, symbol=string.(syms[didx])))
    CSV.write(joinpath(OUT, "parameter_symbols.csv"), DataFrame(index=collect(1:length(ps)), symbol=string.(ps)))
    writedlm(joinpath(OUT, "M_full.csv"), M, ','); writedlm(joinpath(OUT, "A_full.csv"), Afull, ','); writedlm(joinpath(OUT, "A_reduced.csv"), Ared, ','); writedlm(joinpath(OUT, "C_pmu.csv"), C, ',')
    buses = [3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]
    Bout = Dict{Int,Vector{Float64}}(); rows = DataFrame(candidate_bus=Int[], p_symbol=String[], q_symbol=String[], p_index=Int[], q_index=Int[], p_value=Float64[], q_value=Float64[], B_norm=Float64[], status=String[])
    pflat0 = pflat(s0)
    for b in buses
        psym = VIndex(b, :ZIPLoad₊Pset); qsym = VIndex(b, :ZIPLoad₊Qset)
        ip = only(SII.parameter_index(nw, psym)); iq = only(SII.parameter_index(nw, qsym))
        pval = float(pflat0[ip]); qval = float(pflat0[iq])
        # Request all native state variables as outputs so the reduced B column
        # is retained in the same 114-state coordinate system as Ared.
        rp = reduce_dae(linearize_network(s0; in=psym, out=syms)); rq = reduce_dae(linearize_network(s0; in=qsym, out=syms))
        bp = vec(Matrix(rp.B)[:,1]); bq = vec(Matrix(rq.B)[:,1]); bg = pval .* bp .+ qval .* bq
        # Parameter changes also move algebraic outputs instantaneously.  Keep
        # this direct feedthrough term; omitting it would erase the DAE
        # algebraic jump at the event onset.
        lp = reduce_dae(linearize_network(s0; in=[psym, qsym], out=pout))
        dg = vec(Matrix(lp.D) * [pval, qval])
        Bout[b] = bg
        push!(rows, (b,string(psym),string(qsym),ip,iq,pval,qval,norm(bg),all(isfinite,bg) ? "PASS" : "FAIL"))
        writedlm(joinpath(OUT, "B_g_$(b).csv"), bg, ',')
        writedlm(joinpath(OUT, "D_g_$(b).csv"), dg, ',')
    end
    CSV.write(joinpath(OUT, "B_g_map.csv"), rows)
    writedlm(joinpath(OUT, "C_pmu.csv"), C, ',')
    open(joinpath(OUT, "export_status.txt"), "w") do io
        println(io, "ok=true"); println(io, "full_dim=", size(M,1)); println(io, "reduced_dim=", size(Ared,1)); println(io, "differential=", length(didx)); println(io, "algebraic=", length(syms)-length(didx)); println(io, "pmu_dim=", size(C,1)); println(io, "candidate_count=", length(buses))
    end
catch e
    ok = false; err = sprint(showerror, e, catch_backtrace())
    open(joinpath(OUT, "export_status.txt"), "w") do io; println(io, "ok=false"); println(io, err); end
end
println("event_tangent_export_ok=", ok)
!ok && println(err)
