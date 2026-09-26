using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, CairoMakie
using LinearAlgebra, CSV, DataFrames, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", "..")); const OUT = joinpath(ROOT, "output")
const EXAMPLE = joinpath(pkgdir(PowerDynamics), "docs", "examples")

function fit_power(eps, err)
    keep = findall((err .> 1e-12) .& isfinite.(err))
    length(keep) < 3 && return NaN
    x = log.(eps[keep]); y = log.(err[keep]);
    sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)).^2)
end

function run_case(nw, s0, desc, red, didx, aidx, direction, epsv; parameter_index=nothing, parameter_symbol="", tspan=(0.0, 0.25))
    x0 = uflat(s0); p0 = pflat(s0); A = red.A
    rows = DataFrame(scenario=String[], epsilon=Float64[], error=Float64[], error_over_epsilon=Float64[], error_over_epsilon2=Float64[], retcode=String[], finite=Bool[])
    for eps in epsv
        if isnothing(parameter_index)
            vd = direction
            vf = zeros(Float64, length(x0)); vf[didx] = vd; vf[aidx] = -(desc.A[aidx,aidx] \ (desc.A[aidx,didx] * vd))
            xinit = x0 + eps * vf
            prob = ODEProblem(nw, NWState(nw, xinit, p0, 0.0), tspan)
            sol = solve(prob, Rodas5P(); abstol=1e-10, reltol=1e-10)
            maxerr = 0.0
            for k in eachindex(sol.t)
                δlin = eps * (exp(A * sol.t[k]) * vd)
                δnl = sol.u[k][didx] - x0[didx]
                maxerr = max(maxerr, norm(δnl - δlin) / max(1.0, norm(δlin)))
            end
        else
            ppert = copy(p0); ppert[parameter_index] += eps
            prob = ODEProblem(nw, NWState(nw, x0, ppert, 0.0), tspan)
            sol = solve(prob, Rodas5P(); abstol=1e-10, reltol=1e-10)
            # Rebuild the mathematically matching reduced parameter response.
            psym = direction
            lsys = linearize_network(s0; in=psym, out=VIndex(1, :busbar₊u_r))
            lred = reduce_dae(lsys); B = lred.B[:,1]
            n = size(A,1); aug = zeros(Float64, n+1, n+1); aug[1:n,1:n] .= A; aug[1:n,n+1] .= B
            maxerr = 0.0
            for k in eachindex(sol.t)
                z = exp(aug * sol.t[k]) * vcat(zeros(Float64,n), eps)
                δnl = sol.u[k][didx] - x0[didx]
                maxerr = max(maxerr, norm(δnl - z[1:n]) / max(1.0, norm(z[1:n])))
            end
        end
        finite = all(isfinite, sol.u[end]); rc = string(sol.retcode)
        push!(rows, (parameter_index === nothing ? "modal_state" : parameter_symbol, eps, maxerr, maxerr/eps, maxerr/eps^2, rc, finite))
    end
    rows
end

function main()
    ok=true; err=""
    try
        redirect_stdout(devnull) do
            include(joinpath(EXAMPLE,"ieee39_part2.jl"))
        end
        desc = linearize_network(s0); red = reduce_dae(desc); M=Matrix(desc.M); d=diag(M)
        didx=findall(!iszero,d); aidx=findall(iszero,d); vals,V=eigen(red.A)
        physical = findall(real.(vals) .> -0.1)
        idx = first(filter(i -> abs(vals[i]) > 1e-7, physical))
        vd = real(V[:,idx]); vd ./= norm(vd)
        epsv = [1e-2,3e-3,1e-3,3e-4,1e-4,3e-5]
        allrows = run_case(nw,s0,desc,red,didx,aidx,vd,epsv)
        ps = SII.parameter_symbols(nw)
        govidx = findfirst(x -> occursin("p_ref", lowercase(string(x))), ps)
        loadidx = findfirst(x -> occursin("pset", lowercase(string(x))), ps)
        if !isnothing(govidx)
            gs=ps[govidx]; gi=only(SII.parameter_index(nw,gs)); allrows=vcat(allrows,run_case(nw,s0,desc,red,didx,aidx,gs,epsv;parameter_index=gi,parameter_symbol=string(gs)))
        end
        if !isnothing(loadidx)
            ls=ps[loadidx]; li=only(SII.parameter_index(nw,ls)); allrows=vcat(allrows,run_case(nw,s0,desc,red,didx,aidx,ls,epsv;parameter_index=li,parameter_symbol=string(ls)))
        end
        CSV.write(joinpath(OUT,"results","pd_linearization_convergence.csv"),allrows)
        open(joinpath(OUT,"reports","pd_linearization_validation.md"),"w") do io
            println(io,"# PowerDynamics first-order linearization validation\n")
            for scen in unique(allrows.scenario)
                r=allrows[allrows.scenario .== scen,:]; p=fit_power(r.epsilon,r.error)
                println(io,"- `",scen,"`: fitted asymptotic exponent p = `",p,"`; finite retcodes = `",all(r.finite),"`." )
            end
            println(io,"\nErrors are evaluated at the nonlinear solver's native timestamps against the package-supported reduced descriptor response. State perturbations enforce the algebraic tangent `δx_a = -A_aa⁻¹A_ad δx_d`; parameter cases use the matching reduced B matrix." )
        end
        # A compact plot is a diagnostic artifact, not an estimator result.
        fig=Figure(size=(900,600)); ax=Axis(fig[1,1],xscale=log10,yscale=log10,xlabel="epsilon",ylabel="max state error",title="PowerDynamics linearization convergence")
        for scen in unique(allrows.scenario)
r=allrows[allrows.scenario .== scen,:]; lines!(ax,r.epsilon,r.error,label=scen)
        end
        axislegend(ax,position=:rb); save(joinpath(OUT,"plots","pd_linearization_convergence.png"),fig)
        open(joinpath(OUT,"results","pd_modal_confirmation.txt"),"w") do io
            println(io,"mode_index=",idx); println(io,"eigenvalue=",vals[idx]); println(io,"physical_modes_Re_ge_-0.1=",length(physical)); println(io,"nonlinear_confirmation=completed in modal_state scenario over t=",0.25,"s")
        end
    catch e
        ok=false; err=sprint(showerror,e,catch_backtrace()); open(joinpath(OUT,"reports","pd_linearization_validation.md"),"w") do io println(io,"# PowerDynamics first-order linearization validation\n\nSTATUS: ERROR\n\n",err) end
    end
    open(joinpath(OUT,"manifests","linear_validation_status.txt"),"w") do io println(io,"ok=",ok); println(io,err) end
    println("linear_validation_ok=",ok); !ok && println(err)
end

main()
