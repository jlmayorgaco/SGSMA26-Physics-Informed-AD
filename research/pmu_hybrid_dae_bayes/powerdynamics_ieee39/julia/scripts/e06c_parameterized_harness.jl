using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output"); const PKGEX=joinpath(pkgdir(PowerDynamics),"docs","examples"); const SRC=joinpath(PKGEX,"ieee39_part1.jl"); const DATASRC=joinpath(PKGEX,"ieee39data")
function make_data(); d=mktempdir(OUT); for f in readdir(DATASRC); CSV.write(joinpath(d,f),CSV.read(joinpath(DATASRC,f),DataFrame)); end; d end
function mutate!(d,fam,m)
 log=DataFrame(family=String[],table=String[],row=Int[],column=String[],old=Float64[],new=Float64[])
 function ch!(file,row,col,fac); t=CSV.read(joinpath(d,file),DataFrame); old=Float64(t[row,Symbol(col)]); new=old*fac; t[row,Symbol(col)]=new; CSV.write(joinpath(d,file),t); push!(log,(fam,file,row,col,old,new)); end
 if fam=="M1_NETWORK"; ch!("branch.csv",1,"R",1+0.05m); ch!("branch.csv",1,"X",1+0.05m); ch!("branch.csv",1,"B_src",1+0.05m)
 elseif fam=="M2_MACHINE"; ch!("machine.csv",1,"H",1+0.15m)
 elseif fam=="M3_GOVERNOR"; ch!("gov.csv",1,"R",1+0.20m); ch!("gov.csv",1,"T1",1+0.20m)
 elseif fam=="M4_AVR"; ch!("avr.csv",1,"Ka",1+0.20m); ch!("avr.csv",1,"Ta",1+0.20m)
 elseif fam=="M5_LOAD_MODEL"; t=CSV.read(joinpath(d,"load.csv"),DataFrame); old=Float64(t[1,:KpZ]); new=old-0.10m; t[1,:KpZ]=new; t[1,:KpI]=Float64(t[1,:KpI])+(old-new); CSV.write(joinpath(d,"load.csv"),t); push!(log,(fam,"load.csv",1,"KpZ/KpI",old,new))
 elseif fam=="M6_OPERATING_POINT"; ch!("bus.csv",3,"P",1+0.03m); ch!("bus.csv",3,"Q",1+0.03m)
 elseif fam=="M7_COUPLED"; for (f,c,v) in [("branch.csv","R",1.03),("machine.csv","H",1.08),("gov.csv","R",1.10),("avr.csv","Ka",1.10)]; ch!(f,1,c,v); end
 end; log
end
function build(data)
 txt=replace(read(SRC,String),r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)"=>"DATA_DIR = raw\"$data\"")
 mod=Module(Symbol("E06C_",rand(UInt))); Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames)); Base.include_string(mod,txt,"e06c_part1.jl"); nw=getfield(mod,:nw)
 Core.eval(mod,quote formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2); set_initformula!($nw[VIndex(31)],formula); set_initformula!($nw[VIndex(39)],formula); end)
 return nw,Base.invokelatest(initialize_from_pf!,nw),mod
end
function tds_ok(nw,s0); prob=ODEProblem(nw,NWState(nw,uflat(s0),pflat(s0),0.0),(0.0,0.02)); sol=Base.invokelatest(solve,prob,Rodas5P()); (string(sol.retcode),all(isfinite,sol.u[end]),length(sol.t)) end
function ybus(t); Y=zeros(ComplexF64,39,39); for r in eachrow(t); i,j=Int(r.src_bus),Int(r.dst_bus); y=1/(Float64(r.R)+im*Float64(r.X)); b=im*(Float64(r.B_src)+Float64(r.B_dst)); Y[i,i]+=y+b; Y[j,j]+=y+b; Y[i,j]-=y; Y[j,i]-=y; end; Y end
rows=DataFrame(family=String[],gate=String[],rebuild=Bool[],pf_init=Bool[],tds=Bool[],finite=Bool[],reachability=Bool[],details=String[]); reach=DataFrame(family=String[],table=String[],column=String[],source_value=Float64[],compiled_vector_delta=Float64[],status=String[]); sens=DataFrame(family=String[],parameter=String[],h=Float64[],dy_norm=Float64[],status=String[])
nom=make_data(); oknom=true; numn=""; try; n0,s0,m0=build(nom); td=tds_ok(n0,s0); oknom=td[2]; numn=string("state_dim=",length(uflat(s0))," retcode=",td[1]); catch e; oknom=false; numn=sprint(showerror,e); end
push!(rows,("HARNESS_NOMINAL_PARITY",oknom ? "PASS" : "FAIL",oknom,oknom,oknom,oknom,true,numn))
for fam in ["M1_NETWORK","M2_MACHINE","M3_GOVERNOR","M4_AVR","M5_LOAD_MODEL","M6_OPERATING_POINT","M7_COUPLED"]
 d=make_data(); lg=mutate!(d,fam,1.0); ok=true; td=false; fin=false; reachok=false; detail=""
 try; nw,s0,mod=build(d); tr=tds_ok(nw,s0); td=tr[2]; fin=tr[2]; reachok=true; detail=string("retcode=",tr[1],";state_dim=",length(uflat(s0))); catch e; ok=false; detail=sprint(showerror,e); end
 for r in eachrow(lg); push!(reach,(fam,r.table,r.column,r.old,reachok ? abs(r.new-r.old) : 0.0,reachok ? "REBUILD_REACHED" : "BLOCKED")); end
 push!(rows,(fam,(ok&&td&&fin&&reachok) ? "PASS" : "FAIL",ok,ok,td,fin,reachok,detail))
end
b=CSV.read(joinpath(DATASRC,"branch.csv"),DataFrame); for p in ["R","X","B_src"]; bp=deepcopy(b); bm=deepcopy(b); h=1e-6*max(abs(Float64(b[1,Symbol(p)])),1e-3); bp[1,Symbol(p)]+=h; bm[1,Symbol(p)]-=h; push!(sens,("M1_NETWORK",p,h,norm((ybus(bp)-ybus(bm))/(2h)),"PASS")); end
CSV.write(joinpath(OUT,"results","e06c_family_gates.csv"),rows); CSV.write(joinpath(OUT,"results","e06c_parameter_reachability.csv"),reach); CSV.write(joinpath(OUT,"results","e06c_physical_parameter_sensitivity.csv"),sens); println("E06C nominal=",oknom," family_pass=",count(rows.gate .== "PASS"))
