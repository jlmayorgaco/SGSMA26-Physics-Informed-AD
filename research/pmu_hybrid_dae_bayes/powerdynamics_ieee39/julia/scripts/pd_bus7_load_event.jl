using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output"); const PKGEX=joinpath(pkgdir(PowerDynamics),"docs","examples"); const SRC=joinpath(PKGEX,"ieee39_part1.jl"); const DATASRC=joinpath(PKGEX,"ieee39data"); const CASEDIR=joinpath(OUT,"results","full_field_event_reconstruction_v1"); mkpath(CASEDIR)
function make_data(); d=mktempdir(OUT); for f in readdir(DATASRC); CSV.write(joinpath(d,f),CSV.read(joinpath(DATASRC,f),DataFrame)); end; d end
function build(data)
 txt=replace(read(SRC,String),r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)"=>"DATA_DIR = raw\"$data\"")
 mod=Module(Symbol("BUS7_",rand(UInt)))
 Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames))
 redirect_stdout(devnull) do
   Base.include_string(mod,txt,"bus7_part1.jl")
 end
 nw=getfield(mod,:nw)
 Core.eval(mod,quote
   formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
   set_initformula!($nw[VIndex(31)],formula)
   set_initformula!($nw[VIndex(39)],formula)
 end)
 pfs=Base.invokelatest(solve_powerflow,nw;verbose=false)
 s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pfs,verbose=false)
 nw,s0
end
const OBS=[2,5,6,10,19,22,29,39]
function main()
 d=make_data(); nw,s0=build(d); sy=SII.parameter_symbols(nw); bus7_syms=filter(x->occursin("7",string(x)) && (occursin("Pset",string(x)) || occursin("Qset",string(x))),sy); println("BUS7_PARAMETER_SYMBOLS=",bus7_syms)
 aff=ComponentAffect([], [Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]) do u,p,ctx
   p[Symbol("ZIPLoad₊Pset")] *= 1.10; p[Symbol("ZIPLoad₊Qset")] *= 1.10
 end
 set_callback!(nw[VIndex(7)], PresetTimeComponentCallback([2.0],aff))
 sol=solve(ODEProblem(nw,s0,(0.0,5.0)),Rodas5P();abstol=1e-9,reltol=1e-9,saveat=1/30)
 println("RET=",sol.retcode," N=",length(sol.t))
 # Save complete positive-sequence bus voltage truth; native state stays continuous under parameter callback.
 rows=DataFrame(time=Float64[],bus=Int[],V_re=Float64[],V_im=Float64[],V_mag=Float64[],V_ang=Float64[])
 for (k,t) in enumerate(sol.t), b in 1:39; z=complex(float(sol(t;idxs=VIndex(b,:busbar₊u_r))),float(sol(t;idxs=VIndex(b,:busbar₊u_i)))); push!(rows,(t,b,real(z),imag(z),abs(z),angle(z))); end
 CSV.write(joinpath(CASEDIR,"bus7_full_truth.csv"),rows)
 DataFrame(case_id=["BUS7_LOAD_STEP_PQ10_T2"],event=["LOAD_CHANGE"],bus=[7],onset_s=[2.0],deltaP_fraction=[.10],deltaQ_fraction=[.10],p_pre=[-2.3380000305],q_pre=[-.84],p_post=[-2.5718000336],q_post=[-.924],retcode=[string(sol.retcode)],n_frames=[length(sol.t)] ) |> x->CSV.write(joinpath(CASEDIR,"bus7_event_metadata.csv"),x)
end
main()
