"""Regression checks for scale handling, outcome ties, and conformal ranks."""
import unittest
import numpy as np
from scipy.special import ndtri
from src.evaluation.protocol import (conformal_radius,observed_groups,fit_basin_thresholds,
    assign_basin_groups,integrated_quantile_score,gaussian_crps,gaussian_mixture_crps,scores,project_nonempty)

class RevisionProtocolTests(unittest.TestCase):
    def test_zero_ties_are_not_high_flow(self):
        values=np.array([0.]*80+[1.]*20);basins=np.array(['b']*100)
        groups=observed_groups(values,basins)
        self.assertTrue(np.all(groups[:80]==0));self.assertTrue(np.all(groups[80:]==2))
    def test_prediction_thresholds_are_fixed_by_reference(self):
        reference=np.arange(100.);basins=np.array(['b']*100)
        thresholds=fit_basin_thresholds(reference,basins)
        self.assertEqual(assign_basin_groups([1000.],['b'],thresholds)[0],2)
        self.assertEqual(assign_basin_groups([-1000.],['b'],thresholds)[0],0)
    def test_signed_scores_and_exact_rank(self):
        self.assertEqual(scores(np.array([0.]),np.array([2.]),np.array([1.]))[0],-1.)
        self.assertEqual(conformal_radius(np.arange(20.),0.1),18.)
        self.assertTrue(np.isinf(conformal_radius(np.arange(3.),0.01)))
    def test_iqs_affine_scale_equivariance(self):
        y=np.array([0.2,1.1]);q=np.array([[0.,0.3,1.],[0.5,1.,1.5]]);t=[0.05,0.5,0.95]
        base=integrated_quantile_score(y,q,t)
        np.testing.assert_allclose(integrated_quantile_score(y*0.64+0.6,q*0.64+0.6,t),base*0.64)
    def test_dense_integral_matches_analytic_gaussian(self):
        taus=np.linspace(0.00001,0.99999,10001);y=np.array([-1.,0.,1.]);q=np.tile(ndtri(taus),(3,1))
        np.testing.assert_allclose(integrated_quantile_score(y,q,taus),gaussian_crps(y,0.,1.),atol=1e-6)
    def test_mixture_reduces_to_single_gaussian(self):
        y=np.array([-2.,0.,1.]);means=np.zeros((3,2));scales=np.ones((3,2));weights=np.full((3,2),0.5)
        np.testing.assert_allclose(gaussian_mixture_crps(y,means,scales,weights),gaussian_crps(y,0.,1.),atol=1e-12)
    def test_empty_cqr_projection_has_nonnegative_width(self):
        lo,hi,n=project_nonempty([2.,0.],[1.,3.])
        self.assertEqual(n,1);np.testing.assert_allclose(lo,[1.5,0.]);np.testing.assert_allclose(hi,[1.5,3.])

if __name__=='__main__':unittest.main()
