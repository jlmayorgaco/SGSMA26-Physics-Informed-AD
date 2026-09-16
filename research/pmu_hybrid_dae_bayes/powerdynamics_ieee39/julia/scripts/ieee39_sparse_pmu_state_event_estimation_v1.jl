# IEEE39-SPARSE-PMU-STATE-EVENT-ESTIMATION-V1 truth producer.
# Same validated production model and callback as the previous integration
# pilot, with a separate output namespace and no estimator access to truth.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, Random, SHA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "ieee39_sparse_pmu_state_event_estimation_v1", "truth")
const DATA = joinpath(ROOT, "output", "t120_multi_op_independent_validation_v1", "op_data", "op_m085")
const SRC0 = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")
const TAU = 2.0; const DT = 1/30; const TEND = 6.0; const AMP = 0.0033
mkpath(OUT)

function build()
    txt=replace(read(SRC0,String), r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)" => "DATA_DIR = raw\"$DATA\"")
    mod=Module(Symbol("SPARSE_E2E_",rand(UInt)))
    Core.eval(mod, :(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames))
    redirect_stdout(devnull) do
        Base.include_string(mod, txt, "sparse_e2e_ieee39.jl")
    end
    nw=getfield(mod,:nw)
    Core.eval(mod, quote
        formula=@initformula :ZIPLoad₊Vset=sqrt(:busbar₊u_r^2+:busbar₊u_i^2)
        set_initformula!($nw[VIndex(31)],formula); set_initformula!($nw[VIndex(39)],formula)
    end)
    pf=Base.invokelatest(solve_powerflow,nw;verbose=false)
    s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pf,verbose=false)
    nw,s0
end

function callback!(nw,a)
    initialized=Ref(false); p0=Ref(0.0); q0=Ref(0.0); pars=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
    affect=ComponentAffect([],pars) do u,p,ctx
        if !initialized[]; p0[]=Float64(p[pars[1]]); q0[]=Float64(p[pars[2]]); initialized[]=true; end
        p[pars[1]]=p0[]*(1+a); p[pars[2]]=q0[]*(1+a)
    end
    set_callback!(nw[VIndex(7)],PresetTimeComponentCallback([TAU],affect))
end

function one_run(label,a)
    nw,s0=build(); callback!(nw,a); t0=time()
    sol=solve(ODEProblem(nw,deepcopy(s0),(0.0,TEND)),Rodas5P();abstol=1e-11,reltol=1e-11,dtmax=1/60,maxiters=10^7,saveat=DT)
    string(sol.retcode)=="Success" || error("$(label): $(sol.retcode)")
    n=length(uflat(s0)); nt=length(sol.t); umat=Matrix{Float64}(undef,nt,n)
    vr=DataFrame(scenario=String[],time_s=Float64[],sample_index=Int[],bus=Int[],V_re=Float64[],V_im=Float64[])
    for k in 1:nt
        umat[k,:].=Float64.(sol.u[k])
        for b in 1:39
            z=complex(Float64(sol(sol.t[k];idxs=VIndex(b,:busbar₊u_r))),Float64(sol(sol.t[k];idxs=VIndex(b,:busbar₊u_i))))
            push!(vr,(label,Float64(sol.t[k]),k,b,real(z),imag(z)))
        end
    end
    st=DataFrame(umat,Symbol.("u" .* string.(1:n))); insertcols!(st,1,:sample_index=>collect(1:nt)); insertcols!(st,1,:time_s=>Float64.(sol.t)); insertcols!(st,1,:amplitude=>fill(a,nt)); insertcols!(st,1,:scenario=>fill(label,nt))
    sp=joinpath(OUT,"truth_full_state_"*lowercase(label)*".csv.gz"); vp=joinpath(OUT,"truth_full_bus_outputs_"*lowercase(label)*".csv.gz")
    CSV.write(sp,st;compress=true); CSV.write(vp,vr;compress=true)
    (scenario=label,amplitude=a,runtime_s=time()-t0,retcode=string(sol.retcode),n_time=nt,n_state=n,state_path=sp,bus_path=vp,state_sha256=bytes2hex(sha256(read(sp))),bus_sha256=bytes2hex(sha256(read(vp))))
end

rows=[one_run("H0",0.0),one_run("BUS7_EVENT",AMP)]
CSV.write(joinpath(OUT,"truth_execution.csv"),DataFrame(rows))
println("ieee39_sparse_pmu_truth_done")
