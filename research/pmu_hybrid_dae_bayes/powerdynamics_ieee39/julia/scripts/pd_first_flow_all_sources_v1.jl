# Expand the validated production-map recorder to all 16 self directions and
# all 120 unique cross directions.  Richardson remains the reference.
const SRC0 = joinpath(@__DIR__, "pd_t120_first_flow_state_derivative_v1.jl")
txt = read(SRC0, String)
txt = replace(txt,
    "output\", \"t120_first_flow_state_derivative_v1\"" =>
    "output\", \"first_flow_hessian_closure_v1\"")
txt = replace(txt,
    "const SELF_BUSES = [7,26,3,16,12]" =>
    "const SELF_BUSES = [3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]")
txt = replace(txt,
    "const CROSS_PAIRS = [(7,12),(26,28),(3,18),(16,18)]" =>
    "const CROSS_PAIRS = [(i,j) for i in [3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28], j in [3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28] if i < j]")
# Run one operating point per process; files remain resumable/skipped.
old = "for (tag,m) in OPS; run_op(tag,m); end"
new = "for (tag,m) in OPS; only=get(ENV, \"FIRST_FLOW_OP\", \"\"); (isempty(only) || only==tag) && run_op(tag,m); end"
txt = replace(txt, old => new)
# Preserve the source filename so @__DIR__ resolves to this script directory
# (the initial resumable run was launched with an include-string filename and
# therefore wrote to the Julia working directory).  Existing external maps
# remain valid provenance and are copied by the Python audit.
Base.include_string(Main, txt, SRC0)
