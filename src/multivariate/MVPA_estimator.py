from zpyhelper.MVPA.estimators import MetaEstimator, PatternCorrelation, MultipleRDMRegression
from zpyhelper.MVPA.rdm import compute_rdm,compute_rdm_identity,compute_rdm_residual, lower_tri
from zpyhelper.MVPA.preprocessors import split_data,scale_feature
import numpy
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import itertools
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
                 source_ref_split:numpy.ndarray,
                 compose_features:numpy.ndarray,control_features:numpy.ndarray,
                 compose_feature_names:list=None,
                 control_feature_names:list=None,
                 controlrdms_src:Union[numpy.ndarray,list]=[],
                 controlrdms_ref:Union[numpy.ndarray,list]=[],
                 controlrdms_src_name:Union[numpy.ndarray,list]=[],
                 session:numpy.ndarray=[],
                 stimgroup:numpy.ndarray=[],
                 compose_method="concat") -> None:
        """CompositionalRSA analysis, compare the RDM of data and the RDM generated using composed pattern from reference data

        Parameters
        ----------
        activitypattern : numpy.ndarray
            activity pattern matrix, a 2D numpy array of shape ``(nsample,nvoxel)``
        source_ref_split : numpy.ndarray
            whether each row of activity pattern is source (0) or split (1), a 1D numpy array of shape ``(nsample,)`, must contain [0,1]
        compose_features : numpy.ndarray
            stimuli features that are used to match the reference and the source
        control_features : numpy.ndarray
            stimuli features that are used to further constrain the matching
        compose_feature_names : list, optional
            names of compose features, by default None
        control_feature_names : list, optional
            names of control features, by default None
        controlrdms_src : Union[numpy.ndarray,list], optional
            control rdms for data rdm, by default []
        controlrdms_ref : Union[numpy.ndarray,list], optional
            control rdms for composed rdm, by default [] \n
            if `len(controlrdms_ref)>0`, rsa analysis will run on the residuals of composed rdm after regressing out the control rdms
        session : numpy.ndarray, optional
            session of stimuli, a 1D numpy array of shape ``(nsample,)`, must contain [0,1], by default []
        stimgroup : numpy.ndarray, optional
            groups of stimuli, a 1D numpy array of shape ``(nsample,)`, must contain [0,1], by default []
        """
        
        activitypattern  = numpy.atleast_2d(activitypattern)
        compose_features = numpy.atleast_2d(compose_features)
        control_features = numpy.atleast_2d(control_features)        
        assert activitypattern.shape[0] == compose_features.shape[0]
        assert numpy.array_equal(numpy.unique(source_ref_split),[0,1])

        # split into source and ref
        source_mat,    ref_mat    = split_data(activitypattern,groups=source_ref_split,select_groups=[0,1])
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

        # separate and combined feature
        cmbf_apm = dict(zip(
            [f"cpsdrdm_{''.join(self.compose_feature_names)}"],
            [compose_pattern_from_reference(source_cmpsfs,ref_mat,ref_cmpsfs,source_ctrlfs,ref_ctrlfs,compose_method=compose_method)]
        ))
        if self.n_compose_features>1:
            sepf_apm = dict(zip(
                [f"cpsdrdm_{x}" for x in self.compose_feature_names],
                [compose_pattern_from_reference(source_cmpsfs[:,[j]],ref_mat,ref_cmpsfs[:,[j]],source_ctrlfs,ref_ctrlfs) for j in range(self.n_compose_features)]
            ))            
        else:
            sepf_apm = []
        self.composed_apm = dict(zip(list(cmbf_apm.keys()) + list(sepf_apm.keys()), list(cmbf_apm.values()) + list(sepf_apm.values())))
        
        self.composed_info = {
            "stim_session":session,
            "stim_group":stimgroup
        }
        self.control_rdms = {
            "ref":controlrdms_ref,
            "src":controlrdms_src
        }
        self.models = self._construct_compostional_modelrdm(session,stimgroup,controlrdms_ref)

        # define estimator
        #ESTIMATOR_CATELOGUE = {"correlation":PatternCorrelation,"regression":MultipleRDMRegression}
        #estimator_class = "correlation"

        combined_mname = f"cpsdrdm_{''.join(self.compose_feature_names)}"
        modelnames = [combined_mname,f"teststimpairs_{combined_mname}",f"mixedstimpairs_{combined_mname}"]
        modelnames = modelnames + [f"between_{m}" for m in modelnames] + [f"within_{m}" for m in modelnames]
        modelnames = modelnames + [f"resid_{m}" for m in modelnames]
        # filter out those that actually exists
        modelnames = [m for m in modelnames if m in self.models.keys()]
        modelrdms  = [self.models[m] for m in modelnames]      
                
        if len(controlrdms_src)>0:
            self.estimator = []
            for mn,m in zip(modelnames,modelrdms): 
                predictor_lists = [m]  + controlrdms_src
                predictor_names = [mn] + [f"{mn}_ctrl_{x}" for x in controlrdms_src_name]
                predictors = numpy.vstack([lower_tri(x)[0] for x in predictor_lists])
                nan_filter = numpy.all(~numpy.isnan(predictors),0)
                keep_ps = [numpy.unique(arr[nan_filter]).size>1 for arr in predictors]
                if sum(keep_ps)>0:
                    mrdms = [p for p,kp in zip(predictor_lists,keep_ps) if kp]    
                    mrdm_names = [pn for pn,kp in zip(predictor_names,keep_ps) if kp]    
                    self.estimator.append(
                        MultipleRDMRegression(
                            source_mat,
                            modelrdms  = mrdms,
                            modelnames = mrdm_names,
                            rdm_metric="correlation")
                        )
        else:
            self.estimator = [PatternCorrelation(
                                    source_mat,
                                    modelrdms=modelrdms,
                                    modelnames=modelnames,
                                    runonresidual=True,controlrdms=controlrdms_ref,
                                    type="spearman",
                                    rdm_metric="correlation")]
        
    def _construct_compostional_modelrdm(self,session,stimgroup,controlrdms_ref):
        models = dict(zip(self.composed_apm.keys(),
                          [compute_rdm(apm,"correlation") for apm in self.composed_apm.values()]))

        # split rdm into train/test/mix pairs
        if numpy.unique(stimgroup).size>1:
            U,V = numpy.meshgrid(stimgroup,stimgroup)
            WTR = numpy.multiply(1.*(U == 1),1.*(V == 1))
            WTE = numpy.multiply(1.*(U == 0),1.*(V == 0))
            MIX = 1. * ~(U==V)
            WTR[WTR==0]=numpy.nan
            WTE[WTE==0]=numpy.nan
            MIX[MIX==0]=numpy.nan
            split_models = [x for x in models.keys() if x not in ["stimuli","stimuligroup"]]
            for k in split_models:
                wtr_n = 'trainstimpairs_' + k
                rdmwtr = numpy.multiply(models[k],WTR)
                wte_n = 'teststimpairs_' + k
                rdmwte = numpy.multiply(models[k],WTE)
                #mix_n = 'mixedstimpairs_' + k
                #rdmmix = numpy.multiply(models[k],MIX)
                models |= {wtr_n:rdmwtr,wte_n:rdmwte}# ,mix_n:rdmmix

        # within and between session
        if numpy.unique(session).size>1:
            BS = compute_rdm_identity(session) # 0 - within session; 1 - within session
            WS = 1 - BS         # 0 - between session; 1 - between session
            BS[BS==0]=numpy.nan
            WS[WS==0]=numpy.nan
            wsbs_models = {}
            for k,v in models.items():
                ws_n  = 'within_'+k
                rdmws = numpy.multiply(v,WS)
                bs_n  = 'between_'+k
                rdmbs = numpy.multiply(v,BS)
                wsbs_models |= {ws_n:rdmws,bs_n:rdmbs}#
            models.update(wsbs_models)

        # control for control rdm
        if len(controlrdms_ref)>0:
            resid_models = {}
            for mn,m in models.items():
                # resid_m = deepcopy(m)
                # for cm in controlrdms_ref:      
                #     try:
                #         resid_m = compute_rdm_residual(resid_m,cm,squareform=True)
                #     except:
                #         pass
                # resid_models[f"resid_{mn}"] = resid_m
                resid_models[f"resid_{mn}"] = compute_rdm_residual(m,controlrdms_ref,squareform=True)
            models.update(resid_models)
        # remove models in which lower tri have only one value that is not nan
        valid_mn = [k for k,v in models.items() if numpy.sum(~numpy.isnan(numpy.unique(lower_tri(v)[0])))>1]
        valid_m = [models[k] for k in valid_mn]
        models = dict(zip(valid_mn,valid_m))
        
        return models

    def fit(self):
        self.result = numpy.hstack([e.fit().result for e in self.estimator])
        self.resultnames = numpy.hstack([e.modelnames for e in self.estimator])
        return self
    
    def visualize(self):
        fig = self.estimator.visualize()
        return fig
    
    def __str__(self) -> str:
        #if self.compose_separate:
        #    return f"CompositionalRSA based on separate features using {self.estimator.__str__()}"
        #else:
        return f"CompositionalRSA based on all features using {self.estimator[0].__str__()}"
    
    def get_details(self)->str:
        details = {
            "name": self.__str__(),
            "resultnames": list(self.resultnames),
            "estimator": self.estimator[0].get_details()
        }
        return details