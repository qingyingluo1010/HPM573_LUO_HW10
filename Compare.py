import ParameterClasses as P
import MarkovModel as MarkovCls
import SupportMarkovModel as SupportMarkov


# simulating mono therapy
# create a cohort
cohort_NoTreat = MarkovCls.Cohort(
    id=0,
    therapy=P.Therapies.NONE)
# simulate the cohort
simOutputs_NoTreat = cohort_NoTreat.simulate()

# simulating combination therapy
# create a cohort
cohort_Treat = MarkovCls.Cohort(
    id=0,
    therapy=P.Therapies.ANTICOAG)
# simulate the cohort
simOutputs_Treat = cohort_Treat.simulate()

# draw survival curves and histograms
SupportMarkov.draw_survival_curves_and_histograms(simOutputs_NoTreat, simOutputs_Treat)

# print the estimates for the mean survival time and mean time to AIDS
SupportMarkov.print_outcomes(simOutputs_NoTreat, "No Treat:")
SupportMarkov.print_outcomes(simOutputs_Treat, "Treat:")

# print comparative outcomes
SupportMarkov.print_comparative_outcomes(simOutputs_NoTreat, simOutputs_Treat)

# report the CEA results
SupportMarkov.report_CEA_CBA(simOutputs_NoTreat, simOutputs_Treat)