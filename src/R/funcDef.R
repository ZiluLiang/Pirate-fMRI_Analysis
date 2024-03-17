loadRData <- function(fileName){
  #loads an RData file, and returns it
  load(fileName)
  get(ls()[ls() != "fileName"])
}

change_cols_name = function(x,oldnames,newnames){
  if(length(oldnames)!=length(newnames)){
    stop("Length of old names vector and new names vector must match!")
  } else{
    n = length(oldnames)
    for(i in seq(1,n,1)){
      oldname = oldnames[[i]]
      newname = newnames[[i]]
      colnames(x)[colnames(x)==oldname]=newname
    }
  }
  return(x)
}

aggregate_data = function(data,groupbyvars=NULL,yvars,additionalvars=NULL,
                          stats=c("mu","sd","se","sum","n","max","min")){
  stat_funcs = lapply(stats,function(x){
    switch (x,
            "n"   = function(vec){return(sum(!is.na(vec)))},
            "sum" = function(vec){return(sum(vec,na.rm = T))},
            "mu"  = function(vec){return(mean(vec,na.rm = T))},
            "sd"  = function(vec){return(sd(vec,na.rm = T))},
            "se"  = function(vec){return(sd(vec,na.rm = T)/sqrt(sum(!is.na(vec))))},
            "ci95_l"  = function(vec){return(quantile(vec,0.05)[[1]])},
            "ci95_u"  = function(vec){return(quantile(vec,0.95)[[1]])},
            "min"  = function(vec){return(min(vec,na.rm = T))},
            "max"  = function(vec){return(max(vec,na.rm = T))},
            "median"  = function(vec){return(median(vec,na.rm = T))},
            "max"  = function(vec){return(max(vec,na.rm = T))},
    )
  })
  names(stat_funcs) = stats
  
  for(yvar in yvars){
    for(stat in stats){
      newvar = paste0(stat,"_",yvar)
      data = data%>%
          group_by(across(all_of(groupbyvars)))%>%
          mutate({{newvar}}:=stat_funcs[[stat]](.data[[yvar]]))%>%
          ungroup()
      
    }
  }
  
  #newvars = do.call(c,lapply(stats,function(x){return(paste(x,yvars,sep="_"))}))
  newvars = paste0(rep(stats,each=length(yvars)),"_",rep(yvars,length(stats)))
  keepvars = unique(c(groupbyvars,additionalvars,newvars))
  aggdata = data%>%
    distinct(across(all_of(groupbyvars)),.keep_all = TRUE)%>%
    dplyr::select(all_of(keepvars))
  
  return(aggdata)
}


fit_regression = function(data,formula_str,reg_type,...){
  model = switch(reg_type,
                 "lm"    = stats::lm(formula(formula_str),data=data,...),
                 "glm"   = stats::glm(formula(formula_str),data=data, family = binomial,...),
                 "lmer"  = lme4::lmer(formula(formula_str),data=data,...),
                 "glmer" = lme4::glmer(formula(formula_str),
                                 data = data,
                                 family = binomial,
                                 nAGQ=0,...)
                 )
  return(model)
}

tab_regression = function(model,...){
  sjPlot::tab_model(model, use.viewer = TRUE,transform = NULL,dv.labels = "",...)
}


fit_reg_rdm = function(datamodel_rdm_df,f.str,spec_vars,reg_type = "lm",...){
  specs = datamodel_rdm_df[1,spec_vars]
  
  m = fit_regression(data = datamodel_rdm_df,
                     formula_str = f.str,
                     reg_type = reg_type,...)
  
  coef = as.data.frame(m[["coefficients"]])
  coef$RDM = rownames(coef)
  rownames(coef)=NULL
  colnames(coef)[1]="Estimates"
  coef.df = cbind(coef,
                  specs[rep(seq_len(nrow(specs)), each = nrow(coef)), ])
  output = list(coef = coef.df,model = m)
  return(output)
}



