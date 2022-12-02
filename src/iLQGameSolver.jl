module iLQGameSolver
using StaticArrays

greet() = print("Hello World!")

include("foo.jl")
include("cost.jl")
include("linearize_dynamics.jl")
include("lqgame.jl")
include("solveilqgame.jl")
include("dynamics.jl")
include("solveilqgameRH.jl")
# include("recedingHorizon.jl")
# include("closedloop.jl")


end # module
