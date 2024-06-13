from zpyhelper.MVPA.estimators import MetaEstimator, PatternCorrelation, MultipleRDMRegression
from zpyhelper.MVPA.rdm import compute_rdm,compute_rdm_identity,compute_rdm_residual, lower_tri
from zpyhelper.MVPA.preprocessors import split_data,scale_feature
import numpy
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import itertools
import sys
import os
project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'src'))
from multivariate.modelrdms import ModelRDM

from sklearn.linear_model import LinearRegression

def _compose_pattern_from_reference_single_source(source_feature:Union[numpy.ndarray,list],
                                                  R:numpy.ndarray,
                                                  M:numpy.ndarray,
                                                  source_controlfeature:Union[numpy.ndarray,list]=[],
                                                  M0:numpy.ndarray=[],
                                                  compose_method = "concat")->numpy.ndarray:
    """compose pattern for a single source from the reference matrix by matching reference features with the source's features. \n
    Let ``R`` be the reference pattern matrix that contains representation patterns of different feature combinations in the same or a different task.
    Let ``f(x,y)`` be a row vector in ``R`` that correspond to the representation pattern of feature combination represented by vector``(x,y)`` in the voxel space. \n
    We can compose the representation pattern of new feature combination ``(a,b)`` from ``R`` by:\n
      1) selecting rows in ``R`` that contains feature ``a``, same for ``b`` \n
      2) averaging the representation patterns of ``a``, same for ``b`` \n
      3) concatenating the average representation patterns of ``a`` and ``b`` \n
    In other words:    ``g(a,b) = [f(x,y|x=a),f(x,y|y=b)]``   \n
    where ``(a,b)`` is the source feature vector. the reference features matrix ``M`` stores all the information about the features of each rows of ``R``. \n

    In addition, pattern selection further constrained by control features:
    Given ``source_controlfeature``=(c,...), the search of patterns in ``R`` is further limited to make sure only rows where M0 == c is included in the construction of compositional patterns
    
    Parameters
    ----------
    source_feature : Union[numpy.ndarray,list]
        features of the source stimulus that are used to find the corresponding rows in the reference matrix to compose pattern. size =`n`
    R : numpy.ndarray
        the reference pattern matrix. a 2D numpy array that has ``m`` rows. Each row is the representation pattern of a reference stimuli with features specified in the corresponding row of ``M``
    M : numpy.ndarray
        the reference features matrix, a 2D numpy array of shape ``(m,n)`` Each row is the features of a reference stimuli 
    source_controlfeature : Union[numpy.ndarray,list], optional
        control features, by default []
    M0 : numpy.ndarray, optional
        the reference control features matrix specifying control features of the reference samples, by default []

    Returns
    -------
    numpy.ndarray
        composed feature matrix
    """

    source_feature = numpy.array(source_feature).flatten()
    source_controlfeature = numpy.array(source_controlfeature).flatten()
    R = numpy.array(R)
    M = numpy.array(M)
    M0 = numpy.array(M0)

    assert source_feature.size == M.shape[1]
    assert R.shape[0] == M.shape[0]
    if source_controlfeature.size>0:
        assert R.shape[0] == M0.shape[0]
        assert source_controlfeature.size == M0.shape[1]
        #the rows in the reference stimuli where the control feature matches the sample
        ctrl_f_filter = [numpy.array_equal(ref_ctrlfs, source_controlfeature) for ref_ctrlfs in M0]
    else:
         ctrl_f_filter = [True] * R.shape[0]
    
    
    f_filters = [
        [all([x==f,cf]) for x,cf in zip(M[:,j],ctrl_f_filter)] for j,f in enumerate(source_feature)
    ]

    vecs = [numpy.mean(R[f_filter,:],axis=0).flatten() for f_filter in f_filters]
    if compose_method == "concat":
        return numpy.concatenate(vecs)
    elif compose_method == "vec_add":
        return numpy.sum(vecs,axis=0)

def compose_pattern_from_reference(source_features:numpy.ndarray,
                                   reference_pattern:numpy.ndarray,reference_features:numpy.ndarray,
                                   source_controlfeatures:numpy.ndarray=[],reference_controlfeatures:numpy.ndarray=[],
                                   compose_method="concat"):
    """	compose pattern from the reference matrix by matching reference features with the sources' features. \n
    Parameters
    ----------
    source_features : Union[numpy.ndarray,list]
        features of k source stimuli that are used to find the corresponding rows in the reference matrix to compose pattern. a 2D numpy array of shape `(k,n)`
    reference_pattern : numpy.ndarray
        the reference pattern matrix. a 2D numpy array that has ``m`` rows. Each row is the representation pattern of a reference stimuli with features specified in the corresponding row of ``M``
    reference_features : numpy.ndarray
        the reference features matrix, a 2D numpy array of shape ``(m,n)`` Each row is the features of a reference stimuli 
    source_controlfeatures : Union[numpy.ndarray,list], optional
        control features,a 2D numpy array of shape `(k,p)`, by default []
    reference_controlfeatures : numpy.ndarray, optional
        the reference control features matrix specifying control features of the reference samples,a 2D numpy array of shape `(m,p)`, by default []

    Returns
    -------
    numpy.ndarray
        composed feature matrix

    """
    
    source_features        = numpy.atleast_2d(source_features)
    source_controlfeatures = numpy.atleast_2d(source_controlfeatures)
    if source_controlfeatures.size>0:
        assert source_controlfeatures.shape[0] == source_features.shape[0]
        return numpy.array([_compose_pattern_from_reference_single_source(sf,reference_pattern,reference_features,scf,reference_controlfeatures,compose_method=compose_method) for sf,scf in zip(source_features,source_controlfeatures)])
    else:
        return numpy.array([_compose_pattern_from_reference_single_source(sf,reference_pattern,reference_features,compose_method=compose_method) for sf in source_features])


class CompositionalRSA(MetaEstimator):
    def __init__(self,activitypattern:numpy.ndarray,
                 stim_df:pandas.DataFrame,
                 source_ref_split_name:numpy.ndarray,
                 compose_feature_names:list=None,
                 control_feature_names:list=None) -> None:
        """CompositionalRSA analysis, compare the RDM of data and the RDM generated using composed pattern from reference data

        Parameters
        ----------
        activitypattern : numpy.ndarray
            activity pattern matrix, a 2D numpy array of shape ``(nsample,nvoxel)``
        stim_df : pandas.DataFrame
            dataframe containing stimuli information
        source_ref_split_name : numpy.ndarray
            whether each row of activity pattern is source (0) or split (1), a 1D numpy array of shape ``(nsample,)`, must contain [0,1]
        compose_feature_names : list, optional
            names of compose features, by default None
        control_feature_names : list, optional
            names of control features, by default None
        """
        
        activitypattern  = numpy.atleast_2d(activitypattern)
        check_cols = [source_ref_split_name]+compose_feature_names+control_feature_names+["stim_x","stim_y","stim_id","stim_color","stim_shape"]
        assert all([x in stim_df.columns for x in check_cols])
        compose_features,control_features,source_ref_split = stim_df[compose_feature_names].to_numpy(),stim_df[control_feature_names].to_numpy(),stim_df[source_ref_split_name].to_numpy()
        assert activitypattern.shape[0] == compose_features.shape[0]
        assert numpy.array_equal(numpy.unique(source_ref_split),[0,1])

        # split into source and ref
        source_mat, ref_mat = split_data(activitypattern,groups=source_ref_split,select_groups=[0,1])
        source_cmpsfs, ref_cmpsfs = split_data(compose_features,groups=source_ref_split,select_groups=[0,1])
        if control_features.size>0:
            assert activitypattern.shape[0] == control_features.shape[0]
            source_ctrlfs, ref_ctrlfs = split_data(control_features,groups=source_ref_split,select_groups=[0,1])
            self.n_control_features = source_ctrlfs.shape[1]
        else:
            source_ctrlfs, ref_ctrlfs = [],[]
            self.n_control_features = 0

        self.source    = {"activity_pattern":source_mat,
                          "compose_features":source_cmpsfs,
                          "control_features":source_ctrlfs}
        self.reference = {"activity_pattern":ref_mat,
                          "compose_features":ref_cmpsfs,
                          "control_features":ref_ctrlfs}

        self.n_sample = source_mat.shape[0]
        self.n_compose_features = source_cmpsfs.shape[1]        
        
        self.compose_feature_names = [f"{k}" for k in range(self.n_compose_features)] if compose_feature_names is None else compose_feature_names
        assert len(self.compose_feature_names) == self.n_compose_features
        self.control_feature_names = [f"{k}" for k in range(self.n_control_features)] if control_feature_names is None else control_feature_names
        assert len(self.control_feature_names) == self.n_control_features

        
        self.composed_apm = compose_pattern_from_reference(source_cmpsfs,ref_mat,ref_cmpsfs,source_ctrlfs,ref_ctrlfs,compose_method="vec_add")
        self.src_composed_apm = numpy.vstack([source_mat,self.composed_apm])
        self.src_composed_df  = pandas.concat(
            [stim_df[source_ref_split==0].copy().assign(dstype=0),stim_df[source_ref_split==0].copy().assign(dstype=0).assign(dstype=1)],axis=0).reset_index(drop=True)

        submodelrdm = ModelRDM(
                    stimid    = self.src_composed_df["stim_id"].to_numpy(),
                    stimgtloc = self.src_composed_df[["stim_x","stim_y"]].to_numpy(),
                    stimfeature = self.src_composed_df[["stim_color","stim_shape"]].to_numpy(),
                    stimgroup = self.src_composed_df["stim_group"].to_numpy(),
                    sessions =  self.src_composed_df["dstype"].to_numpy(),
                    nan_identity = False,
                    splitgroup  = False
                )
        correlation_config={"euclidean":"between_gtlocEuclidean","feature":"between_feature2d","stimuli":"between_stimuli"}
        self.modelRDMs = dict(zip(correlation_config.keys(),[submodelrdm.models[m] for m in correlation_config.values()]))


        self.PCestimator = PatternCorrelation(
            activitypattern=self.src_composed_apm,
            modelnames=list(self.modelRDMs.keys()),
            modelrdms=list(self.modelRDMs.values()),
            rdm_metric="correlation",
            type="spearman",
            ztransform=False
        )

    def fit(self):
        self.result = self.PCestimator.fit().result
        self.resultnames = list(self.modelRDMs.keys())
        return self
    
    def visualize(self):
        fig = self.estimator.visualize()
        return fig
    
    def __str__(self) -> str:
        #if self.compose_separate:
        #    return f"CompositionalRSA based on separate features using {self.estimator.__str__()}"
        #else:
        return f"CompositionalRSA with {self.PCestimator.__str__()}"
    
    def get_details(self)->str:
        details = {
            "name": self.__str__(),
            "resultnames": list(self.resultnames),
            "estimator": self.PCestimator.get_details()
        }
        return details