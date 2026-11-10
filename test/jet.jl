@testitem "JET" begin
    using NonconvexMMA, Test
    # JET 0.13's `report_package` corrupts the loaded method tables on
    # Julia 1.13 (its virtual top-level replay re-evaluates `@params`
    # definitions in the running process), so analyze in a fresh subprocess
    # to keep the test process intact for the remaining testitems.
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    project = dirname(Base.active_project())
    code = "using JET, NonconvexMMA\n" *
           "JET.test_package(NonconvexMMA; target_modules=(NonconvexMMA,))"
    @test success(
        pipeline(`$julia --startup-file=no --project=$project -e $code`; stdout=devnull, stderr=devnull)
    )
end
