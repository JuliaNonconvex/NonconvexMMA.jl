using SafeTestsets

@safetestset "MMA approximation" begin
    include("approximation.jl")
end
@safetestset "MMA algorithms" begin
    include("mma.jl")
end

@safetestset "Aqua" begin
    include("aqua.jl")
end
@safetestset "JET" begin
    include("jet.jl")
end
