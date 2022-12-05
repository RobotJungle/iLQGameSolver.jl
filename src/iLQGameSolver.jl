module iLQGameSolver
using StaticArrays

greet() = print("Hello World!")

include("foo.jl")
include("quadraticizeCost.jl")
include("linearizeDynamics.jl")
include("solveilqGame.jl")
include("lqGame.jl")
include("gameDynamics.jl")

# include("recedingHorizon.jl")
# include("closedloop.jl")


end # module
