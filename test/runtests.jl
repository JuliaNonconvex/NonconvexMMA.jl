using SafeTestsets

@safetestset "MMA approximation" begin
    include("approximation.jl")
end
@safetestset "MMA algorithms" begin
    include("mma.jl")
end
@safetestset "Solver capabilities" begin
    include("capabilities.jl")
end
