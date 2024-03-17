packagelist = c("tidyverse","ggtext","ggsignif","lme4","lmerTest",
                "emmeans","multcomp","margins","cowplot","ggh4x",
                "pracma","multcomp","jtools","huxtable","broom.mixed",
                "doParallel","parallel","foreach","rstatix","car",
                "gridExtra","pbapply","Cairo")

installlist = packagelist[which(!packagelist %in% rownames(installed.packages()))]
lapply(installlist, install.packages)

lapply(packagelist,function(x){
  library(x,character.only=TRUE)
})

