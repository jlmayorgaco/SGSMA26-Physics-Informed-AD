using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random, SHA

const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output"); const PKGEX=joinpath(pkgdir(PowerDynamics),"docs","examples"); const SRC=joinpath(PKGEX,"ieee39_part1.jl"); const DATASRC=joinpath(PKGEX,"ieee39data"); const E06E_MODE=haskey(ENV,"E06E_MODE"); const E06F_MODE=haskey(ENV,"E06F_MODE"); const E06G_MODE=haskey(ENV,"E06G_MODE"); const E06E_TAG=get(ENV,"E06E_TAG",""); const E06F_TAG=get(ENV,"E06F_TAG",""); const E06G_TAG=get(ENV,"E06G_TAG","m6_diagnostic_v1"); const CASEDIR=joinpath(OUT,"results",E06E_MODE ? (isempty(E06E_TAG) ? "e06e_cases" : "e06e_cases_"*E06E_TAG) : (E06F_MODE ? (isempty(E06F_TAG) ? "pf_recenter_cases" : "pf_recenter_cases_"*E06F_TAG) : (E06G_MODE ? "e06g_cases_"*E06G_TAG : "e06_standard_cases")))
mkpath(CASEDIR); mkpath(joinpath(OUT,"results")); mkpath(joinpath(OUT,"reports"))
const FAMILIES=["M1_NETWORK","M2_MACHINE","M3_GOVERNOR","M4_AVR","M5_LOAD_MODEL","M6_OPERATING_POINT","M7_COUPLED"]; const LEVELS=[0.0,0.25,0.5,0.75,1.0,1.25,1.5]; const EXC=["E-A","E-B","E-C","E-D"]
function make_data(); d=mktempdir(OUT); for f in readdir(DATASRC); CSV.write(joinpath(d,f),CSV.read(joinpath(DATASRC,f),DataFrame)); end; d end
function mutate!(d,fam,m)
    log=String[]
    function ch!(file,row,col,fac); t=CSV.read(joinpath(d,file),DataFrame); old=Float64(t[row,Symbol(col)]); new=old*fac; t[row,Symbol(col)]=new; CSV.write(joinpath(d,file),t); push!(log,string(file,":",row,":",col,":",old,"->",new)); end
    if fam=="M1_NETWORK"; ch!("branch.csv",1,"R",1+0.05m); ch!("branch.csv",1,"X",1+0.05m); ch!("branch.csv",1,"B_src",1+0.05m)
    elseif fam=="M2_MACHINE"; ch!("machine.csv",1,"H",1+0.15m)
    elseif fam=="M3_GOVERNOR"; ch!("gov.csv",1,"R",1+0.20m); ch!("gov.csv",1,"T1",1+0.20m)
    elseif fam=="M4_AVR"; ch!("avr.csv",1,"Ka",1+0.20m); ch!("avr.csv",1,"Ta",1+0.20m)
    elseif fam=="M5_LOAD_MODEL"; t=CSV.read(joinpath(d,"load.csv"),DataFrame); old=Float64(t[1,:KpZ]); new=old-0.10m; t[1,:KpZ]=new; t[1,:KpI]=Float64(t[1,:KpI])+(old-new); CSV.write(joinpath(d,"load.csv"),t); push!(log,string("load.csv:1:KpZ/KpI:",old,"->",new))
    elseif fam=="M6_OPERATING_POINT"; ch!("bus.csv",3,"P",1+0.03m); ch!("bus.csv",3,"Q",1+0.03m)
    elseif fam=="M7_COUPLED"; for (f,c,v) in [("branch.csv","R",1+0.03m),("machine.csv","H",1+0.08m),("gov.csv","R",1+0.10m),("avr.csv","Ka",1+0.10m)]; ch!(f,1,c,v); end
    end; log
end
function build(data)
    txt=replace(read(SRC,String),r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)"=>"DATA_DIR = raw\"$data\"")
    mod=Module(Symbol("E06S_",rand(UInt))); Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames))
    redirect_stdout(devnull) do; Base.include_string(mod,txt,"e06_standard_part1.jl"); end; nw=getfield(mod,:nw)
    Core.eval(mod,quote formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2); set_initformula!($nw[VIndex(31)],formula); set_initformula!($nw[VIndex(39)],formula); end)
    pfs=Base.invokelatest(solve_powerflow,nw;verbose=false); s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pfs,verbose=false); nw,s0,pfs
end
const OBS=[2,5,6,10,19,22,29,39]; const HIDDEN=setdiff(1:39,OBS); const EDGE_FOR=[11,37,18,35,31,1,8,5]; const NATIVE_FOR=[false,false,true,true,true,false,false,false]
function channels(); p=Any[]; h=Any[]; for b in OBS; push!(p,VIndex(b,:busbar₊u_r)); push!(p,VIndex(b,:busbar₊u_i)); end; for (e,n) in zip(EDGE_FOR,NATIVE_FOR); s=n ? :src : :dst; push!(p,EIndex(e,Symbol(s,"₊i_r"))); push!(p,EIndex(e,Symbol(s,"₊i_i"))); end; for b in HIDDEN; push!(h,VIndex(b,:busbar₊u_r)); push!(h,VIndex(b,:busbar₊u_i)); end; p,h end
function vals(sol,t,p,h); y=[float(sol(t;idxs=q)) for q in p]; z=[float(sol(t;idxs=q)) for q in h]; y,z end
function trajectory(nw,s0,seed,exc)
    x0=uflat(s0); p0=pflat(s0); sy=SII.parameter_symbols(nw); pp=copy(p0)
    # Keep excitation amplitudes on the preregistered 1..20 schedule even
    # when the independent split uses disjoint seed identifiers.
    sseed=mod(seed-1,20)+1
    function bump!(needles,frac); j=findfirst(x->any(occursin(n,lowercase(string(x))) for n in needles),sy); if !isnothing(j); pp[only(SII.parameter_index(nw,sy[j]))]*=(1+frac); return true; end; false end
    if exc=="E-A"; bump!(["pset"],0.015+0.001sseed)
    elseif exc=="E-B"; bump!(["p_ref"],0.015+0.001sseed)
    elseif exc=="E-C"; bump!(["q_ref","qset","v_ref","vset"],0.012+0.001sseed)
    else; bump!(["p_ref"],0.010+0.0005sseed); bump!(["pset"],0.010+0.0005sseed); end
    sol=Base.invokelatest(solve,ODEProblem(nw,NWState(nw,x0,pp,0.0),(0.0,3.0)),Rodas5P();abstol=1e-9,reltol=1e-9); sol
end
function case_id(f,m,seed,e); replace(f,"_"=>"-")*"_m"*replace(string(m),"."=>"p")*"_s"*string(seed)*"_"*replace(e,"-"=>"") end
function write_case(cid,f,m,seed,e)
    mp=joinpath(CASEDIR,cid*"_meta.csv"); tp=joinpath(CASEDIR,cid*"_trajectory.csv"); isfile(mp)&&isfile(tp)&&return "CHECKPOINT"
    d=make_data(); lg=mutate!(d,f,m); ph=bytes2hex(sha1(reduce(vcat,[read(joinpath(d,x)) for x in sort(readdir(d))])))
    try
        nw,s0,pfs=build(d); p,h=channels(); sol=trajectory(nw,s0,seed,e); ts=collect(range(0.0,3.0,length=91)); rows=DataFrame(case_id=String[],frame=Int[],time=Float64[]); for k in 1:32; rows[!,Symbol("pmu_",k)]=Float64[]; end; for k in 1:62; rows[!,Symbol("hidden_",k)]=Float64[]; end
        for (k,t) in enumerate(ts); y,z=vals(sol,t,p,h); push!(rows,(cid,k,t,y...,z...)); end; CSV.write(tp,rows)
        meta=DataFrame(case_id=[cid],family=[f],m=[m],seed=[seed],excitation=[e],plant_hash=[ph],pf_status=["PASS"],init_status=["PASS"],tds_status=[string(sol.retcode)],finite=[all(isfinite,sol.u[end])],metric_status=["PENDING"],mutation_manifest=[join(lg,";")]); CSV.write(mp,meta); "ACCEPT"
    catch err
        reason=occursin("powerflow",lowercase(sprint(showerror,err))) ? "PF_FAIL" : occursin("init",lowercase(sprint(showerror,err))) ? "DYNAMIC_INIT_FAIL" : "TDS_FAIL"; meta=DataFrame(case_id=[cid],family=[f],m=[m],seed=[seed],excitation=[e],plant_hash=[ph],pf_status=[reason=="PF_FAIL" ? "FAIL" : "UNKNOWN"],init_status=[reason=="DYNAMIC_INIT_FAIL" ? "FAIL" : "UNKNOWN"],tds_status=[reason=="TDS_FAIL" ? "FAIL" : "UNKNOWN"],finite=[false],metric_status=[reason],mutation_manifest=[join(lg,";")]); CSV.write(mp,meta); "REJECT_"*reason
    end
end
manifest=DataFrame(case_id=String[],family=String[],m=Float64[],seed=Int[],excitation=String[],kind=String[])
if E06G_MODE
    # E06-G uses a fresh M6-only diagnostic split: 10 seeds per scale.
    fams=["M6_OPERATING_POINT"]
    e06glevels=[0.5,1.0,1.5]
    seeds=501:510
    for f in fams, m in e06glevels, seed in seeds; e=EXC[mod1(seed,length(EXC))]; push!(manifest,(case_id(f,m,seed,e),f,m,seed,e,"DIAGNOSTIC")); end
    CSV.write(joinpath(OUT,"results","e06g_manifest.csv"),manifest)
elseif E06F_MODE
    fams=["M1_NETWORK","M2_MACHINE","M6_OPERATING_POINT","M7_COUPLED"]
    e06flevels=[0.0,0.5,1.0,1.5]
    testseeds=haskey(ENV,"E06F_SPLIT") && ENV["E06F_SPLIT"]=="TEST"; seeds=testseeds ? (401:420) : (301:310)
    for f in fams, m in e06flevels, seed in seeds; e=EXC[mod1(seed,length(EXC))]; push!(manifest,(case_id(f,m,seed,e),f,m,seed,e,haskey(ENV,"E06F_SPLIT") ? ENV["E06F_SPLIT"] : "DEV")); end
    CSV.write(joinpath(OUT,"results",testseeds ? "e06f_test_manifest.csv" : "e06f_dev_manifest.csv"),manifest)
elseif E06E_MODE
    # Independent trajectory seeds: 101-110 for DEV and 201-220 for TEST.
    # They are deliberately disjoint from E04/E06 STANDARD/E06-D seeds.
    fams=["M1_NETWORK","M2_MACHINE","M6_OPERATING_POINT","M7_COUPLED"]
    e06levels=[0.0,0.5,1.0,1.5]
    testseeds=haskey(ENV,"E06E_SPLIT") && ENV["E06E_SPLIT"]=="TEST"
    seeds=testseeds ? (201:220) : (101:110)
    for f in fams, m in e06levels, seed in seeds; e=EXC[mod1(seed,length(EXC))]; push!(manifest,(case_id(f,m,seed,e),f,m,seed,e,haskey(ENV,"E06E_SPLIT") ? ENV["E06E_SPLIT"] : "DEV")); end
    CSV.write(joinpath(OUT,"results",testseeds ? "e06e_test_manifest.csv" : "e06e_dev_manifest.csv"),manifest)
elseif haskey(ENV,"E06_STANDARD_REFINED")
    bands=Dict("M1_NETWORK"=>[0.75,0.875,1.0,1.125,1.25],"M2_MACHINE"=>[1.0,1.125,1.25,1.375,1.5],"M3_GOVERNOR"=>[0.75,0.875,1.0,1.125,1.25],"M4_AVR"=>[0.75,0.875,1.0,1.125,1.25],"M5_LOAD_MODEL"=>[0.5,0.625,0.75,0.875,1.0],"M6_OPERATING_POINT"=>[0.5,0.625,0.75,0.875,1.0],"M7_COUPLED"=>[0.25,0.375,0.5,0.625,0.75])
    for f in FAMILIES, m in bands[f], seed in 21:25; e=EXC[mod1(seed,length(EXC))]; push!(manifest,(case_id(f,m,seed,e),f,m,seed,e,"refined")); end
    CSV.write(joinpath(OUT,"results","e06_standard_refined_manifest.csv"),manifest)
else
    for f in FAMILIES, m in LEVELS, seed in 1:20; e=EXC[mod1(seed,length(EXC))]; push!(manifest,(case_id(f,m,seed,e),f,m,seed,e,"main_grid")); end
    CSV.write(joinpath(OUT,"results","e06_standard_manifest.csv"),manifest)
end
if haskey(ENV,"E06_STANDARD_ONLY"); manifest=manifest[manifest.case_id .== ENV["E06_STANDARD_ONLY"],:]; end
if haskey(ENV,"E06_STANDARD_FAMILIES"); fs=split(ENV["E06_STANDARD_FAMILIES"],","); manifest=manifest[in.(manifest.family,Ref(fs)),:]; end
for r in eachrow(manifest); cid=r.case_id; status=write_case(cid,r.family,r.m,r.seed,r.excitation); status!="CHECKPOINT" && println(status," ",cid); end
println("E06 STANDARD generation complete: ",nrow(manifest)," cases")
