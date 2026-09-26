module PowerDynamicsBus7LoadPulse

include("Bus7LoadPulse.jl")
using .Bus7LoadPulse

export run_scenario

run_scenario(; kwargs...) = Bus7LoadPulse.run_scenario(; kwargs...)

end
