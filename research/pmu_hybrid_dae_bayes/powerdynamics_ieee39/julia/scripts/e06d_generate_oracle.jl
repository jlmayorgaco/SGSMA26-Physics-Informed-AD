using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random, Dates

const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output")
const PKGEX=joinpath(pkgdir(PowerDynamics),"docs","examples"); const SRC=joinpath(PKGEX,"ieee39_part1.jl")
const DATASRC=joinpath(PKGEX,"ieee39data"); const CASEDIR=joinpath(OUT,"results","e06d_cases")
mkpath(CASEDIR); mkpath(joinpath(OUT,"results")); mkpath(joinpath(OUT,"reports")); mkpath(joinpath(OUT,"plots"))

function make_data()
    d=mktempdir(OUT)
    for f in readdir(DATASRC)
        CSV.write(joinpath(d,f),CSV.read(joinpath(DATASRC,f),DataFrame))
    end
    d
end
function mutate!(d,fam,m)
    log=DataFrame(family=String[],table=String[],row=Int[],column=String[],old=Float64[],new=Float64[])
    function ch!(file,row,col,fac)
        t=CSV.read(joinpath(d,file),DataFrame); old=Float64(t[row,Symbol(col)]); new=old*fac
        t[row,Symbol(col)]=new; CSV.write(joinpath(d,file),t); push!(log,(fam,file,row,col,old,new))
    end
    if fam=="M1_NETWORK"; ch!("branch.csv",1,"R",1+0.05m); ch!("branch.csv",1,"X",1+0.05m); ch!("branch.csv",1,"B_src",1+0.05m)
    elseif fam=="M2_MACHINE"; ch!("machine.csv",1,"H",1+0.15m)
    elseif fam=="M3_GOVERNOR"; ch!("gov.csv",1,"R",1+0.20m); ch!("gov.csv",1,"T1",1+0.20m)
    elseif fam=="M4_AVR"; ch!("avr.csv",1,"Ka",1+0.20m); ch!("avr.csv",1,"Ta",1+0.20m)
    elseif fam=="M5_LOAD_MODEL"
        t=CSV.read(joinpath(d,"load.csv"),DataFrame); old=Float64(t[1,:KpZ]); new=old-0.10m; t[1,:KpZ]=new; t[1,:KpI]=Float64(t[1,:KpI])+(old-new); CSV.write(joinpath(d,"load.csv"),t); push!(log,(fam,"load.csv",1,"KpZ/KpI",old,new))
    elseif fam=="M6_OPERATING_POINT"; ch!("bus.csv",3,"P",1+0.03m); ch!("bus.csv",3,"Q",1+0.03m)
    elseif fam=="M7_COUPLED"; for (f,c,v) in [("branch.csv","R",1+0.03m),("machine.csv","H",1+0.08m),("gov.csv","R",1+0.10m),("avr.csv","Ka",1+0.10m)]; ch!(f,1,c,v); end
    end
    log
end

function build(data)
    txt=replace(read(SRC,String),r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)"=>"DATA_DIR = raw\"$data\"")
    mod=Module(Symbol("E06D_",rand(UInt)))
    Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames))
    redirect_stdout(devnull) do; Base.include_string(mod,txt,"e06d_part1.jl"); end
    nw=getfield(mod,:nw)
    Core.eval(mod,quote formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2); set_initformula!($nw[VIndex(31)],formula); set_initformula!($nw[VIndex(39)],formula); end)
    pfs=Base.invokelatest(solve_powerflow,nw;verbose=false)
    s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pfs,verbose=false)
    nw,s0,pfs,mod
end

const OBS=[2,5,6,10,19,22,29,39]; const HIDDEN=setdiff(1:39,OBS)
const EDGE_FOR=[11,37,18,35,31,1,8,5]; const NATIVE_FOR=[false,false,true,true,true,false,false,false]
function channels()
    pout=Any[]; hout=Any[]
    for b in OBS; push!(pout,VIndex(b,:busbar₊u_r)); push!(pout,VIndex(b,:busbar₊u_i)); end
    for (e,native) in zip(EDGE_FOR,NATIVE_FOR); side=native ? :src : :dst; push!(pout,EIndex(e,Symbol(side,"₊i_r"))); push!(pout,EIndex(e,Symbol(side,"₊i_i"))); end
    for b in HIDDEN; push!(hout,VIndex(b,:busbar₊u_r)); push!(hout,VIndex(b,:busbar₊u_i)); end
    pout,hout
end
function output_values(st,pout,hout)
    iv=interface_values(st); y=Float64[]; h=Float64[]
    for q in pout; push!(y,float(iv[q])); end
    for q in hout; push!(h,float(iv[q])); end
    y,h
end
function solution_values(sol,t,pout,hout)
    y=Float64[]; h=Float64[]
    for q in pout; push!(y,float(sol(t;idxs=q))); end
    for q in hout; push!(h,float(sol(t;idxs=q))); end
    y,h
end
function linearize(nw,s0,pout,hout)
    desc=linearize_network(s0); red=reduce_dae(desc); A=Matrix(red.A)
    lp=reduce_dae(linearize_network(s0;in=VIndex(1,:busbar₊u_r),out=pout)); lh=reduce_dae(linearize_network(s0;in=VIndex(1,:busbar₊u_r),out=hout))
    syms=SII.variable_symbols(nw); d=diag(Matrix(desc.M)); vars=syms[findall(!iszero,d)]
    A,Matrix(lp.C),Matrix(lh.C),string.(vars)
end
function trajectory(nw,s0,mod,seed)
    x0=uflat(s0); p0=pflat(s0); ps=SII.parameter_symbols(nw); j=findfirst(x->occursin("p_ref",lowercase(string(x))),ps)
    pp=copy(p0); amp=0.02*(1+0.15*(seed-1)); !isnothing(j) && (pp[only(SII.parameter_index(nw,ps[j]))]*=(1+amp))
    prob=ODEProblem(nw,NWState(nw,x0,pp,0.0),(0.0,0.50)); sol=Base.invokelatest(solve,prob,Rodas5P();abstol=1e-9,reltol=1e-9)
    sol
end
function write_case(case_id,fam,m,seed)
    d=make_data(); mutate!(d,fam,m); nw,s0,pfs,mod=build(d); pout,hout=channels(); A,C,L,vars=linearize(nw,s0,pout,hout); y0,h0=output_values(pfs,pout,hout); sol=trajectory(nw,s0,mod,seed)
    ts=collect(range(0.0,0.50,length=31)); rows=DataFrame(case_id=String[],frame=Int[],time=Float64[])
    for k in 1:32; rows[!,Symbol("pmu_",k)]=Float64[]; end; for k in 1:62; rows[!,Symbol("hidden_",k)]=Float64[]; end
    for (k,t) in enumerate(ts)
        y,h=solution_values(sol,t,pout,hout); push!(rows,(case_id,k,t,y...,h...))
    end
    CSV.write(joinpath(CASEDIR,case_id*"_trajectory.csv"),rows)
    CSV.write(joinpath(CASEDIR,case_id*"_A.csv"),DataFrame(A,:auto)); CSV.write(joinpath(CASEDIR,case_id*"_C.csv"),DataFrame(C,:auto)); CSV.write(joinpath(CASEDIR,case_id*"_L.csv"),DataFrame(L,:auto))
    CSV.write(joinpath(CASEDIR,case_id*"_y0.csv"),DataFrame(value=y0)); CSV.write(joinpath(CASEDIR,case_id*"_h0.csv"),DataFrame(value=h0));
    CSV.write(joinpath(CASEDIR,case_id*"_state_symbols.csv"),DataFrame(index=collect(1:length(vars)),symbol=vars))
    meta=DataFrame(case_id=[case_id],family=[fam],m=[m],seed=[seed],state_dim=[size(A,1)],pmu_dim=[size(C,1)],hidden_dim=[size(L,1)],retcode=[string(sol.retcode)],finite=[all(isfinite,sol.u[end])],trajectory_frames=[length(ts)])
    CSV.write(joinpath(CASEDIR,case_id*"_meta.csv"),meta)
    println("DONE ",case_id," ",size(A)," ",size(C)," ",size(L))
end

families=["M1_NETWORK","M2_MACHINE","M3_GOVERNOR","M4_AVR","M5_LOAD_MODEL","M6_OPERATING_POINT","M7_COUPLED"]
spec=DataFrame(case_id=String[],family=String[],m=Float64[],seed=Int[],kind=String[])
for fam in families, m in [0.5,1.0,1.5], seed in [1,2]
    cid=replace(fam,"_"=>"-")*"_m"*replace(string(m),"."=>"p")*"_s"*string(seed); push!(spec,(cid,fam,m,seed,"mutated"))
end
for seed in [1,2]
    cid="NOMINAL_m0_s"*string(seed); push!(spec,(cid,"NOMINAL",0.0,seed,"nominal"))
end
CSV.write(joinpath(OUT,"results","e06d_manifest.csv"),spec)
if haskey(ENV,"E06D_ONLY")
    spec=spec[spec.case_id .== ENV["E06D_ONLY"],:]
end
for r in eachrow(spec)
    try; write_case(r.case_id,r.family,r.m,r.seed); catch e; open(joinpath(CASEDIR,r.case_id*"_ERROR.txt"),"w") do io; showerror(io,e,catch_backtrace()); end; println("ERROR ",r.case_id," ",sprint(showerror,e)); end
end
println("E06D generation complete: ",nrow(spec)," cases")
