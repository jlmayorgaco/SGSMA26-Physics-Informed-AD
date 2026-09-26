using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using LinearAlgebra, CSV, DataFrames, Random

const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output")
const EXAMPLE=joinpath(pkgdir(PowerDynamics),"docs","examples")
const OBS=[2,5,6,10,19,22,29,39]; const HIDDEN=setdiff(1:39,OBS)
const edge_for=[11,37,18,35,31,1,8,5]; const native_for=[false,false,true,true,true,false,false,false]

redirect_stdout(devnull) do; include(joinpath(EXAMPLE,"ieee39_part2.jl")); end
x0=uflat(s0); p0=pflat(s0); ps=SII.parameter_symbols(nw)
gi=only(SII.parameter_index(nw,ps[findfirst(x->occursin("p_ref",lowercase(string(x))),ps)]))
li=only(SII.parameter_index(nw,ps[findfirst(x->occursin("pset",lowercase(string(x))),ps)]))

function obsval(sol,t)
    z=Float64[]
    for b in OBS
        push!(z,float(sol(t;idxs=VIndex(b,:busbar₊u_r)))); push!(z,float(sol(t;idxs=VIndex(b,:busbar₊u_i))))
    end
    for (e,native) in zip(edge_for,native_for)
        if native
            push!(z,float(sol(t;idxs=EIndex(e,:src₊i_r)))); push!(z,float(sol(t;idxs=EIndex(e,:src₊i_i))))
        else
            push!(z,float(sol(t;idxs=EIndex(e,:dst₊i_r)))); push!(z,float(sol(t;idxs=EIndex(e,:dst₊i_i))))
        end
    end
    z
end
function hidval(sol,t)
    z=Float64[]
    for b in HIDDEN
        push!(z,float(sol(t;idxs=VIndex(b,:busbar₊u_r)))); push!(z,float(sol(t;idxs=VIndex(b,:busbar₊u_i))))
    end
    z
end

const n=30; const root_seed=20262000; const ts=collect(range(0,3,length=91))
function main()
rng=MersenneTwister(root_seed); specs=DataFrame(traj=String[],split=String[],seed=Int[],scenario=String[],amplitude_a=Float64[],amplitude_b=Float64[],parent_seed=Int[])
for j in 1:n
    a=0.002+0.008*rand(rng); b=0.002+0.008*rand(rng)
    typ=mod(j,3)==0 ? "A+B" : mod(j,3)==1 ? "A_governor" : "B_load"
    da=typ=="B_load" ? 0.0 : a; db=typ=="A_governor" ? 0.0 : b
    push!(specs,("CAL_NOMINAL_V1_$(j)","CAL_NOMINAL_V1",root_seed+j,typ,da,db,20260911))
end
mkpath(joinpath(OUT,"results")); mkpath(joinpath(OUT,"manifests"))
CSV.write(joinpath(OUT,"results","e04a3_cal_split_manifest.csv"),specs)

dataset=DataFrame(); y0=nothing; h0=nothing
for (row,j) in zip(eachrow(specs),1:n)
    pp=copy(p0); pp[gi]*=(1+row.amplitude_a); pp[li]*=(1+row.amplitude_b)
    sol=solve(ODEProblem(nw,NWState(nw,x0,pp,0.0),(0.0,3.0)),Rodas5P();abstol=1e-9,reltol=1e-9)
    finite=all(isfinite,sol.u[end]); rc=string(sol.retcode)
    if j==1
        s0sol=solve(ODEProblem(nw,NWState(nw,x0,p0,0.0),(0.0,0.01)),Rodas5P();abstol=1e-9,reltol=1e-9)
        y0=obsval(s0sol,0.0); h0=hidval(s0sol,0.0)
    end
    for (k,t) in enumerate(ts)
        ov=obsval(sol,t); hv=hidval(sol,t)
        vals=Any[row.traj,row.split,row.scenario,row.seed,row.amplitude_a,row.amplitude_b,k-1,t,finite,rc]
        append!(vals,ov); append!(vals,hv)
        names=Symbol.(vcat(["traj","split","scenario","seed","amplitude_a","amplitude_b","frame","time","finite","retcode"], ["pmu_$(i)" for i in 1:length(ov)], ["hidden_$(i)" for i in 1:length(hv)]))
        one=DataFrame(); for (nm,val) in zip(names,vals); one[!,nm]=[val]; end
        dataset=vcat(dataset,one;cols=:union)
    end
end
CSV.write(joinpath(OUT,"results","e04a3_cal_dataset.csv"),dataset)
CSV.write(joinpath(OUT,"results","e04a3_cal_y0_pmu.csv"),DataFrame(pmu=y0))
CSV.write(joinpath(OUT,"results","e04a3_cal_hidden0.csv"),DataFrame(hidden=h0))
open(joinpath(OUT,"manifests","e04a3_cal_dataset_status.txt"),"w") do io
    println(io,"ok=true"); println(io,"split=CAL_NOMINAL_V1"); println(io,"trajectories=30"); println(io,"frames=91"); println(io,"duration_s=3.0"); println(io,"root_seed=",root_seed)
end
println("e04a3_cal_dataset_ok=true")
end
main()
