
rm(list = ls())
project_dir = rstudioapi::getActiveProject();
src_dir = file.path(project_dir, "src/R", fsep="/")
llr_dir = file.path(project_dir, "src/LLR", fsep="/")

data_dir = file.path(project_dir, "data","Exp1_fmri", fsep="/")
result_dir = file.path(project_dir, "results","Exp1_fmri", fsep="/")


source(file.path(src_dir,"envSetup.R",fsep="/"), local = knitr::knit_global())
source(file.path(src_dir,"funcDef.R",fsep="/"), local = knitr::knit_global())
source(file.path(src_dir,"setPlotDefault.R",fsep="/"), local = knitr::knit_global())

data_with_prob = loadRData(file.path(data_dir,"filtereddata_withprob_compo_lapse-0_01.RData"))
meanscanperf_with_prob = loadRData(file.path(data_dir,"filteredmeanscannerdata_withprob_compo_lapse-0_01.RData"))


# Split pretrain data into odd trials and even trials
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

# now let's plot and check the error curve for each participant
lapply(group_split(pretrain_traindata,subid), function(subdf){
  sid = unique(subdf$subid)[1]
  sg = unique(subdf$isgeneralizer)[1]
  sl = unique(subdf$islearner)[1]
  if(sl=="learner"&sg=="generalizer"){
    subdf = subdf%>%dplyr::select(c("subid","stim_id","stim_x","stim_y","trainingtrialid","error_x_rescaled","error_y_rescaled"))
    subdflong = subdf%>%
      pivot_longer(cols=c("error_x_rescaled","error_y_rescaled"),names_to = "axis",names_pattern="error_(.)",values_to = "error")
    p = ggplot(subdflong,aes(x=trainingtrialid,y=error,color=axis))+
      geom_point()+
      labs(x="training trial",y="error",title=paste(sid,sg,sep = "-"))
    ggsave(file.path(result_dir,'inidividualerrcurve',paste0('trainerror',sid,'.png')),
           plot = p,device = 'png',width = 6,height = 6)
  }
})
lapply(group_split(pretrain_testdata,subid), function(subdf){
  sid = unique(subdf$subid)[1]
  sg = unique(subdf$isgeneralizer)[1]
  sl = unique(subdf$islearner)[1]
  if(sl=="learner"&sg=="generalizer"){
    subdf = subdf%>%dplyr::select(c("subid","stim_id","stim_x","stim_y","testtrialid","error_x_rescaled","error_y_rescaled"))
    subdflong = subdf%>%
      pivot_longer(cols=c("error_x_rescaled","error_y_rescaled"),names_to = "axis",names_pattern="error_(.)",values_to = "error")
    p = ggplot(subdflong,aes(x=testtrialid,y=error,color=axis))+
      geom_point()+
      labs(x="test trial",y="error",title=paste(sid,sg,sep = "-"))
    ggsave(file.path(result_dir,'inidividualerrcurve',paste0('testerror',sid,'.png')),
           plot = p,device = 'png',width = 6,height = 6)
  }
})

# run the fitting in python

# now lets check results
fit_res_df = read.table(file.path(data_dir,"pretraintrainerrorfit_warmstartgridsearch.csv"), header = TRUE, sep = ",")%>%
  dplyr::select(-c("X"))

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
  geom_point(size=1)+
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
  mutate(asym_cont = abs(odd+even))


ggplot(data=xmid_axisdiff_df_wideforsplit,
       aes(x=odd,y=even,color=subcohort))+
  geom_point(size=1)+
  facet_grid(cols=vars(subgroup))+
  geom_vline(xintercept = 0)+
  geom_hline(yintercept = 0)+
  theme(legend.position="top")+
  labs(color="Cohort",x="difference in error switch point (x-y)\n odd split",y="difference in error switch point (x-y)\n even split")

write_csv(xmid_axisdiff_df_wideforsplit,file=file.path(data_dir,"Asymlearning.csv"))

