from enum import Enum
import numpy as np
import scipy.stats as stat
import math as math
import InputData as Data
import scr.MarkovClasses as MarkovCls
import scr.RandomVariantGenerators as Random
import scr.ProbDistParEst as Est

class HealthStats(Enum):
    """ health states of patients with HIV """
    WELL = 1
    STROKE=2
    POST_STROKE = 3
    DEATH = 3


class Therapies(Enum):
    """ mono vs. combination therapy """
    NONE = 0
    ANTICOAG = 1


class _Parameters:
    def __init__(self, therapy):

        # selected therapy
        self._therapy = therapy

        # simulation time step
        self._delta_t = Data.DELTA_T

        # calculate the adjusted discount rate
        self._adjDiscountRate = Data.DISCOUNT*Data.DELTA_T

        # initial health state
        self._initialHealthState = HealthStats.WELL


        # annual treatment cost
        if self._therapy == Therapies.NONE:
            self._annualTreatmentCost = 0
        else:
            self._annualTreatmentCost = Data.Anticoagulation_Cost

        # transition probability matrix of the selected therapy
        self._prob_matrix = []
        # treatment relative risk
        self._treatmentRR = 0

        # calculate transition probabilities depending of which therapy options is in use
        if therapy == Therapies.NONE:
            self._prob_matrix = Data.TRANS_MATRIX
        else:
            self._prob_matrix = calculate_prob_matrix_anticoag()

        # annual state costs and utilities
        self._annualStateCosts = []
        self._annualStateUtilities = []

    def get_initial_health_state(self):
        return self._initialHealthState

    def get_delta_t(self):
        return self._delta_t

    def get_adj_discount_rate(self):
        return self._adjDiscountRate


    def get_transition_prob(self, state):
        return self._prob_matrix[state.value]

    def get_annual_state_cost(self, state):
        return self._annualStateCosts[state.value]


class ParametersFixed(_Parameters):
    def __init__(self, therapy):

        # initialize the base class
        _Parameters.__init__(self, therapy)

        # calculate transition probabilities between health states
        self._prob_matrix = calculate_prob_matrix_anticoag()

        # update the transition probability matrix if combination therapy is being used
        if self._therapy == Therapies.ANTICOAG:
            # treatment relative risk
            if self._healthstate == HealthStats.POST_STROKE:
                self._treatmentRR = Data.RR_BLEEDING * Data.RR_STROKE
            if self._healthstate == HealthStats.STROKE:
                self._treatmentRR = Data.RR_BLEEDING

        # annual state costs and utilities
        self._annualStateCosts = Data.ANNUAL_STATE_COST
        self._annualStateUtilities = Data.ANNUAL_STATE_UTILITY


class ParametersProbabilistic(_Parameters):
    def __init__(self, seed, therapy):

        # initializing the base class
        _Parameters.__init__(self, therapy)

        self._rng = Random.RNG(seed)    # random number generator to sample from parameter distributions
        self._hivProbMatrixRVG = []  # list of dirichlet distributions for transition probabilities
        self._lnRelativeRiskRVG = None  # random variate generator for the treatment relative risk
        self._annualStateCostRVG = []       # list of random variate generators for the annual cost of states
        self._annualStateUtilityRVG = []    # list of random variate generators for the annual utility of states

        # stroke transition probabilities
        j = 0
        for prob in Data.TRANS_MATRIX:
            self._hivProbMatrixRVG.append(Random.Dirichlet(prob[j:]))
            j += 1

        # treatment relative risk
        # find the mean and st_dev of the normal distribution assumed for ln(RR)
        sample_mean_lnRR = math.log(Data.RR_BLEEDING)
        sample_std_lnRR = (Data.RR_BLEEDING_CI[1]-Data.RR_BLEEDING_CI[0])/(2*stat.norm.ppf(1-0.05/2))
        self._lnRelativeRiskRVG = Random.Normal(mean=sample_mean_lnRR, st_dev=sample_std_lnRR)

        # annual state cost
        for cost in Data.ANNUAL_STATE_COST:
            # find shape and scale of the assumed gamma distribution
            shape, scale = Est.get_gamma_parameters(mean=cost, st_dev=cost/ 4)
            # append the distribution
            self._annualStateCostRVG.append(Random.Gamma(shape, scale))

        # annual state utility
        for utility in Data.ANNUAL_STATE_UTILITY:
            # find alpha and beta of the assumed beta distribution
            a, b = Est.get_beta_parameters(mean=utility, st_dev=utility/5)
            # append the distribution
            self._annualStateUtilityRVG.append(Random.Beta(a, b))

        # resample parameters
        self.__resample()

    def __resample(self):

        # calculate transition probabilities
        # create an empty matrix populated with zeroes
        self._prob_matrix = []
        for s in HealthStats:
            self._prob_matrix.append([0] * len(HealthStats))

        # for all health states
        for s in HealthStats:
            # if the current state is death
            if s in [HealthStats.DEATH]:
                # the probability of staying in this state is 1
                self._prob_matrix[s.value][s.value] = 1
            else:
                # sample from the dirichlet distribution to find the transition probabilities between hiv states
                dist = self._hivProbMatrixRVG[s.value]
                sample = dist.sample(self._rng)
                for j in range(len(sample)):
                    self._prob_matrix[s.value][s.value+j] = sample[j]


        # sample from gamma distributions that are assumed for annual state costs
        self._annualStateCosts = []
        for dist in self._annualStateCostRVG:
            self._annualStateCosts.append(dist.sample(self._rng))

        # sample from beta distributions that are assumed for annual state utilities
        self._annualStateUtilities = []
        for dist in self._annualStateUtilityRVG:
            self._annualStateUtilities.append(dist.sample(self._rng))

def calculate_prob_matrix_anticoag():
    """ :returns transition probability matrix under anticoagulation use"""

    # create an empty matrix populated with zeroes
    prob_matrix = []
    for s in HealthStats:
        prob_matrix.append([0] * len(HealthStats))

    # for all health states
    for s in HealthStats:
        # if the current state is post-stroke
        if s == HealthStats.POST_STROKE:
            # post-stoke to stroke
            prob_matrix[s.value][HealthStats.STROKE.value]\
                = Data.RR_STROKE*Data.TRANS_MATRIX[s.value][HealthStats.STROKE.value]
            # post-stroke to death
            prob_matrix[s.value][HealthStats.DEATH.value] \
                = Data.RR_STROKE * Data.RR_BLEEDING * Data.TRANS_MATRIX[s.value][HealthStats.DEATH.value]
            # staying in post-stroke
            prob_matrix[s.value][s.value]\
                = 1 -prob_matrix[s.value][HealthStats.STROKE.value] -prob_matrix[s.value][HealthStats.DEATH.value]
        else:
            prob_matrix[s.value] = Data.TRANS_MATRIX[s.value]

    return prob_matrix