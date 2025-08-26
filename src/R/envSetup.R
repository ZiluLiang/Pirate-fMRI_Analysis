packagelist = c("tidyverse","ggtext","ggsignif","lme4","lmerTest","ggpubr",
                "emmeans","multcomp","margins","cowplot","ggh4x","grid",
                "pracma","multcomp","jtools","huxtable","broom.mixed","openxlsx",
                "doParallel","parallel","foreach","rstatix","car","corrr",
                "gridExtra","pbapply","Cairo","colorspace")

installlist = packagelist[which(!packagelist %in% rownames(installed.packages()))]
lapply(installlist, install.packages)

lapply(packagelist,function(x){
  library(x,character.only=TRUE)
})

