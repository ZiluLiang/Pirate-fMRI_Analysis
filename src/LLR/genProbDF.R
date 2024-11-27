prob_df_list = list()
pb <- utils::winProgressBar(title = "progress bar", min = 0, max = nrow(unique_stimloc) , width = 300)

for (k in unique_stimloc$loc_id){
  probability_df = data.frame(x = c(cart_GRID$X),y = c(cart_GRID$Y))
  truex = unique_stimloc$x[k]
  truey = unique_stimloc$y[k]

  ##binorm: calculate from approximated bivariate Gaussian distribution
  cl <- makeCluster(detectCores())
  clusterExport(cl=cl,
                varlist = c("gaussian_prob_discret_gridproxy","probability_df","granularity",
                            "truex","truey","arena_radius_rescale","dist_sd"),
                envir=environment())
  p_list <- parLapply(cl, seq(1,nrow(probability_df)),
                      function(j){
                        p = gaussian_prob_discret_gridproxy(arena_radius_rescale,
                                                            probability_df$x[j],probability_df$y[j],
                                                            truex,truey,granularity,sd = dist_sd)
                        return(p)}
  )
  stopCluster(cl)
  p_unlist = unlist(p_list)
  p_unlist_normalized = p_unlist/sum(p_unlist,na.rm = T)
  probability_df[["binorm"]] = p_unlist_normalized

  ##binorm_compo: calculate from the multiplication of two approximated univariate Gaussian distribution
  cl <- makeCluster(detectCores())
  clusterExport(cl=cl,
                varlist = c("gaussian_prob_univariate_grid_proxy","probability_df","granularity",
                            "truex","truey","arena_radius_rescale","dist_sd"),
                envir=environment())
  p_list <- parLapply(cl, seq(1,nrow(probability_df)),
                      function(j){
                        p = gaussian_prob_univariate_grid_proxy(arena_radius_rescale,
                                                                probability_df$x[j],probability_df$y[j],
                                                                truex,truey,granularity,sd = dist_sd)
                        return(p)}
  )
  stopCluster(cl)
  probability_df = cbind(probability_df,do.call(rbind,p_list))
  sum_pnormx = sum(probability_df$pnorm_x,na.rm = T)
  sum_pnormy = sum(probability_df$pnorm_y,na.rm = T)
  
  probability_df = probability_df %>%
    mutate(pnorm_x_normalized = pnorm_x/sum_pnormx,
           pnorm_y_normalized = pnorm_y/sum_pnormy)%>%
    mutate(pnorm_xy = pnorm_x_normalized*pnorm_y_normalized)
  
  sum_pnormxy = sum(probability_df$pnorm_xy,na.rm = T)
  probability_df = probability_df %>%
    mutate(binorm_compo = pnorm_xy/sum_pnormxy)


  ##uniform random
  cl <- makeCluster(detectCores())
  clusterExport(cl=cl,
                varlist = c("uniform_prob_discret","probability_df","granularity",
                            "truex","truey","arena_radius_rescale"),
                envir=environment())
  p_list <- parLapply(cl, seq(1,nrow(probability_df)),
                      function(j){
                        p = uniform_prob_discret(arena_radius_rescale,
                                                 probability_df$x[j],probability_df$y[j],
                                                 truex,truey,granularity)
                        return(p)})
  stopCluster(cl)
  p_unlist = unlist(p_list)
  p_unlist_normalized = p_unlist/sum(p_unlist,na.rm = T)
  probability_df[["uniform"]] = p_unlist_normalized

  probability_df = probability_df%>%
    mutate(truex = truex,truey = truey)
  filename = file.path(llr_dir,"probdf_nolapse_csv",paste0("probdf",k,".csv"))
  write.csv(probability_df, filename, row.names=FALSE)

  prob_df_list = append(prob_df_list, list(probability_df))

  setWinProgressBar(pb, k, title=paste( round(k/nrow(unique_stimloc)*100, 0),"% done"))
}
save(prob_df_list, unique_stimloc, cart_GRID, base, file = file.path(llr_dir,"probDF_data_compo.RData"))
