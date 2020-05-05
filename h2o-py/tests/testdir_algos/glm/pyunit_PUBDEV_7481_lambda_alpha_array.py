import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

# Given arrays of lambda and alpha, the glm should return the best submodel in its training metrics calculation.
#
# When an array of alpha and/or lambdas are given, a list of submodels are also built.  For each submodel built, only
# the coefficients, lambda/alpha/deviance values are returned.  The model metrics is calculated from the submodel
# with the best deviance.  
#
# In this test, in addition, we build separate models using just one lambda and one alpha values as when building one
# submodel.  In theory, the coefficients obtained from the separate models should equal to the submodels.  We check 
# and compare the followings:
# 1. coefficients from submodels and individual model should match when they are using the same alpha/lambda value;
# 2. training metrics from alpha array should equal to the individual model matching the alpha/lambda value.
def glm_alpha_lambda_arrays():
    # read in the dataset and construct training set (and validation set)
    d = h2o.import_file(path=pyunit_utils.locate("smalldata/logreg/prostate.csv"))
    m = glm(family='binomial',Lambda=[0.9,0.5,0.1], alpha=[0.1,0.5,0.9])
    m.train(training_frame=d,x=[2,3,4,5,6,7,8],y=1)
    pyunit_utils.compareSubmodelsNindividualModels(m, d, [2, 3, 4, 5, 6, 7, 8], 1)

if __name__ == "__main__":
    pyunit_utils.standalone_test(glm_alpha_lambda_arrays)
else:
    glm_alpha_lambda_arrays()
