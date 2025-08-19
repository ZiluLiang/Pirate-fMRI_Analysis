
rm(list = ls())
project_dir = rstudioapi::getActiveProject();
src_dir = file.path(project_dir, "src/R", fsep="/")
llr_dir = file.path(project_dir, "src/LLR", fsep="/")

data_dir = file.path(project_dir, "data","Exp1_fmri", fsep="/")
result_dir = file.path(project_dir, "results","Exp1_fmri", fsep="/")


source(file.path(src_dir,"envSetup.R",fsep="/"), local = knitr::knit_global())
source(file.path(src_dir,"funcDef.R",fsep="/"), local = knitr::knit_global())
source(file.path(src_dir,"setPlotDefault.R",fsep="/"), local = knitr::knit_global())



############ Section 1: Split pretrain data into odd trials and even trials
data_with_prob = loadRData(file.path(data_dir,"filtereddata_withprob_compo_lapse-0_01.RData"))
meanscanperf_with_prob = loadRData(file.path(data_dir,"filteredmeanscannerdata_withprob_compo_lapse-0_01.RData"))
pretrain_traindata = filter(data_with_prob,stage=="train"&expt_session==1)
pretrain_testdata = filter(data_with_prob,stage=="test"&expt_session==1)
#check if number of totoal trial is correct, the following must return TRUE
N_traintrial = 9*2*6
N_testtrial = 16*6
nrow(pretrain_traindata) == length(unique(pretrain_traindata$subid)) * N_traintrial
nrow(pretrain_testdata) == length(unique(pretrain_testdata$subid)) * N_testtrial

# when x/y range is -3.4~3.4, radius is five; now rescale radius to x/y range is -1,1
arena_radius_rescale = (5+48/53)*(2/6.8) 
max_error = arena_radius_rescale*2
#generate trial id for each participant
pretrain_traindata = pretrain_traindata%>%
  group_by(subid)%>%
  arrange(expt_trial)%>%
  mutate(trainingtrialid=row_number())%>%
  mutate(trainingtrialid_scaled = trainingtrialid/N_traintrial,
         error_x_rescaled = (error_x-0)/max_error,
         error_y_rescaled = (error_y-0)/max_error)%>%
  ungroup()

pretrain_testdata = pretrain_testdata%>%
  group_by(subid)%>%
  arrange(expt_trial)%>%
  mutate(testtrialid=row_number())%>%
  mutate(testtrialid_scaled = testtrialid/N_testtrial,
         error_x_rescaled = (error_x-0)/max_error,
         error_y_rescaled = (error_y-0)/max_error)%>%
  ungroup()
#check the trialid is correct with:
all.equal(unique(pretrain_traindata$trainingtrialid),seq(1,N_traintrial))
all.equal(unique(pretrain_testdata$testtrialid),seq(1,N_testtrial))

write_csv(pretrain_traindata,file=file.path(data_dir,"trialwisepretraintraindata.csv"))
write_csv(pretrain_testdata,file=file.path(data_dir,"trialwisepretraintestdata.csv"))

############ Section 2: run parameter recovery in python
sigmoidrecovery_res_df = read.table(file.path(data_dir,"sigmoiderror_paramrecovery.csv"), header = TRUE, sep = ",")%>%
  dplyr::select(-c("X"))
asymrecovery_res_df = read.table(file.path(data_dir,"pretrainerror_paramrecovery_softl1loss.csv"), header = TRUE, sep = ",")%>%
  dplyr::select(-c("X"))%>%
  mutate(noiselevel=factor(noiselevel,levels=c("low","medium","high")))
asymrecovery_res_df_wider = asymrecovery_res_df%>%
  mutate(groundtruthrounded=round(groundtruth,2))%>%
  pivot_wider(names_from = param,
              values_from = c(groundtruth,groundtruthrounded, estimates,initguess))

paramhuevars = list("u"="b","s"="u","b"="u")
for (paramname in unique(asymrecovery_res_df$param)){
  print(paramname)
  gt_colname = str_c("groundtruth_",paramname)
  est_colname = str_c("estimates_",paramname)
  x0_colname = str_c("initguess_",paramname)
  color_colname = str_c("groundtruth_",paramhuevars[[paramname]])
  (p = 
      ggplot(data=asymrecovery_res_df_wider,
             aes(x=.data[[gt_colname]],
                 y=.data[[x0_colname]])
             )+
        facet_grid2(cols=vars(noiselevel),rows=vars(datasplit))+
        geom_point(aes(color=.data[[color_colname]]),size=1,alpha=0.6)+
        stat_cor(aes(label=after_stat(r.label)), # show correlation coefficient
                 color = "black", geom = "text",
                 digits=2)+
    
        scale_color_viridis_c()+
        labs(x=str_c("Groudtruth of ",paramname),
             y=str_c("Estimates of ",paramname),
             color=str_c("Groundtruth of ",paramhuevars[[paramname]]),
             subtitle=str_c("Parameter Recovery of ",paramname)
        )+
        theme(legend.position = "top")
    )
}



######## run model fitting in python
# now lets check results
fit_res_df = read.table(file.path(data_dir,"pretraintesterrorfit_uniformgridsearch_softl1loss.csv"), header = TRUE, sep = ",")%>%
  dplyr::select(-c("X"))%>%
  mutate(datasplit=factor(datasplit,levels=c("odd","even")))

fit_res_df_r2 = fit_res_df %>%
  aggregate_data(groupbyvars = c("subid","subgroup","subcohort","datasplit"),
                 yvars = c("r2"),
                 stats = "mu")%>%
  change_cols_name("mu_r2","r2")
fit_res_df_r2_sum = fit_res_df %>%
  aggregate_data(groupbyvars = c("subgroup","subcohort","datasplit"),
                 yvars = c("r2"),
                 stats = c("mu","se"))


ggplot(fit_res_df_r2)+
  facet_grid2(cols=vars(subgroup),scales="free",independent="all")+
  geom_boxplot(aes(x=datasplit, y=r2, color=subcohort),position=position_dodge(1))+
  geom_point(aes(x=datasplit, y=r2, color=subcohort),position=position_jitterdodge(jitter.width=0.2,dodge.width = 1),alpha=0.5)+
  theme(legend.position="top")

ggplot(fit_res_df_r2,
       aes(x=datasplit,colour=subcohort,group=subcohort))+
  facet_grid2(cols=vars(subgroup),scales="free",independent="all")+
  geom_col(data=fit_res_df_r2_sum,
           aes(y=mu_r2),
           position=position_dodge(1),fill=NA)+
  geom_errorbar(data=fit_res_df_r2_sum,
                aes(ymin=mu_r2-se_r2,ymax=mu_r2+se_r2),
                position=position_dodge(1),width=.3)+
  geom_point(aes(y=r2),position=position_jitterdodge(jitter.width=0.2,dodge.width = 1),alpha=0.5)+
  theme(legend.position="top")

fit_res_df_wideforsplit = pivot_wider(fit_res_df%>%dplyr::select(-c("r2")),
                                      id_cols=c("subid","subgroup","subcohort","yvar","param"),
                                      names_from = "datasplit",
                                      values_from = "estimates")



ggplot(data=filter(fit_res_df_wideforsplit,subgroup=="Generalizer"),
       aes(x=odd,y=even,color=subcohort))+
  geom_point()+ #egom_count
  geom_smooth(method="lm")+
  geom_abline(slope=1,intercept=0,colour="black",linetype="dashed")+
  facet_grid2(cols=vars(param),scales="free",independent="all",axes="all",remove_labels=F)+
  theme(legend.position="top")+
  labs(color="Cohort",x="estimates in odd split",y="estimates in even split")

xmid_axisdiff_df = pivot_wider(filter(fit_res_df,param =="b"),#%>%mutate(halflife = exp(2)/estimates),
                                     id_cols=c("param","subid","subgroup","subcohort","datasplit"),
                                     names_from = "yvar",
                                     values_from = "estimates")%>%
  mutate(xmid_axisdiff = error_x_rescaled-error_y_rescaled)

xmid_axisdiff_df_wideforsplit = pivot_wider(xmid_axisdiff_df,
                                            id_cols=c("subid","subgroup","subcohort"),
                                            names_from = "datasplit",
                                            values_from = "xmid_axisdiff")%>%
  mutate(asym = if_else(sign(odd)==sign(even),"asym","sym"))%>%
  mutate(asym_cont = odd+even)

ggplot(data=xmid_axisdiff_df_wideforsplit,
       aes(x=odd,y=even,color=subcohort))+
  geom_point(size=1)+
  facet_grid(cols=vars(subgroup))+
  geom_vline(xintercept = 0)+
  geom_hline(yintercept = 0)+
  theme(legend.position="top")+
  labs(color="Cohort",x="difference in error switch point (x-y)\n odd split",y="difference in error switch point (x-y)\n even split")

ggplot(xmid_axisdiff_df_wideforsplit,
       aes(x=subcohort,color=asym,fill=asym))+
  facet_grid(cols=vars(subgroup))+
    geom_bar(position = position_dodge(0.91))+
  theme(legend.position="top")
write_csv(xmid_axisdiff_df_wideforsplit,file=file.path(data_dir,"Asymlearning.csv"))

