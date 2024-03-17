generate_RDM = function(data, splitby_var, sortby_var, similarity_var, specs_var){
  #subset data by participant-block
  data.subset = data%>%group_split(across(all_of(splitby_var)))
  
  # sub-level rdm df with specs
  ##generate complete matrix for plotting purpose
  data.subset = lapply(data.subset,function(df) df%>%arrange(across(all_of(sortby_var))))
  
  # use parrallel apply to generate df
  cl <- makeCluster(detectCores())
  clusterExport(cl=cl, 
                varlist = c("create_rdmdf","create_matrix","rdm2dataframe","mat2dataframe","specs_var","similarity_var"),
                envir=environment())
  data.similarity.sub <- parLapply(cl, data.subset, function(df) create_rdmdf(df,similarity_var,specs_var))
  stopCluster(cl)
  
  tmp = do.call(rbind,data.similarity.sub) 
  data.similarity.sub = tmp
  rm(tmp)
  
  # ##generate lower tri matrix for conducting reg analysis
  # data.similarity.sub.tri = lapply(data.subset,function(df) create_halfrdmdf(df,similarity_var,specs_var)) #for each sub subset do this
  # tmp = do.call(rbind,data.similarity.sub.tri) #for each block subset do this
  # data.similarity.sub.tri = tmp
  
  # df.similarity = list(full = data.similarity.sub, tri = data.similarity.sub.tri)
  return(data.similarity.sub)
}

create_matrix = function(x){
  grid = pracma::meshgrid(x)
  mat = 1*(grid$X == grid$Y)
  vec = as.integer(grid$X == grid$Y)
  x_pos = c(0,1,2)
  matrix = list("matrix" = mat,"vector"=vec)
  return(mat)
}

mat2dataframe = function(mat,nvalperdim = 3){
  colnames(mat) = as.character(seq(1,nvalperdim^3,1))
  rownames(mat) = as.character(seq(1,nvalperdim^3,1))
  df = suppressWarnings(reshape::melt.array(mat))
  return(df)
}

rdm2dataframe = function(rdm,nvalperdim = 3){
  df = mat2dataframe(rdm,nvalperdim)
  colnames(df) = c("x", "y", "similarity")
  
  #indexing upper tri and lower tri
  loc_mat = 1*upper.tri(rdm)+(-1)*lower.tri(rdm)
  loc_df = mat2dataframe(loc_mat,nvalperdim)
  colnames(loc_df) = c("x", "y", "tri_loc")
  loc_df$tri_loc = factor(loc_df$tri_loc,levels = c(-1,0,1),labels = c("lower","diag","upper"))
  df = dplyr::left_join(df,loc_df,by=c("x","y"))
  return(df)
}

create_rdmdf = function(data,similarity_var,specs_vars){
  mat = create_matrix(data[[similarity_var]])
  df = rdm2dataframe(mat)
  specs = data[1,specs_vars]
  specs = specs[rep(seq_len(nrow(specs)), each = nrow(df)), ]
  df = cbind(specs,df)
  return(df)
}


halfrdm2dataframe = function(rdm){
  df = cbind(which(!is.na(rdm),arr.ind = TRUE),na.omit(as.vector(rdm)))
  df = as.data.frame(df)
  colnames(df) = c("x", "y", "similarity")
  return(df)
}

create_halfrdmdf = function(data,similarity_var,spec_vars){
  mat = create_matrix(data[[similarity_var]])
  mat[upper.tri(mat)] <- NA
  df = halfrdm2dataframe(mat)
  specs = data[1,spec_vars]
  specs = specs[rep(seq_len(nrow(specs)), each = nrow(df)), ]
  df = cbind(specs,df)
  return(df)
}

mean_matrix = function(matrix.list){
  n = length(matrix.list)
  mean = Reduce('+',matrix.list)/n
}