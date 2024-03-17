gaussian_prob_discret_gridproxy = function(r,respx,respy,truex,truey,granularity=265*2,sd = 0.1){
  #r - radius of response arena
  base = seq(from = -r,to = r,length.out = granularity)
  xlb = base[max(which(base<=respx))]
  xub = base[min(which(base>respx))]
  ylb = base[max(which(base<=respy))]
  yub = base[min(which(base>respy))]
  
  outside_arena = sqrt(xlb^2+ylb^2)>r & sqrt(xub^2+yub^2)>r & sqrt(xub^2+ylb^2)>r & sqrt(xlb^2+yub^2)>r
  if (outside_arena){
    p = NA
  } else{
    InnerFunc = function(x,y) { dnorm(x,truex,sd = sd)*dnorm(y,truey,sd = sd)}
    InnerIntegral = Vectorize(function(y) {
      #ub = ifelse(xub>0,min(xub,sqrt(r^2-y^2)),max(xub,-sqrt(r^2-y^2)))
      #lb = ifelse(xlb>0,min(xub,sqrt(r^2-y^2)),max(xlb,-sqrt(r^2-y^2)))
      integrate(function(x){InnerFunc(x,y)},
                xlb,#max(xlb,-sqrt(r^2-y^2)),
                xub#min(xub,sqrt(r^2-y^2))
                )$value
    })
    p = integrate(InnerIntegral, ylb,yub)$value
  }
  return(p)
}


uniform_prob_discret = function(r,respx,respy,truex,truey,granularity=265*2){
  base = seq(from = -r,to = r,length.out = granularity)
  xlb = base[max(which(base<=respx))]
  xub = base[min(which(base>respx))]
  ylb = base[max(which(base<=respy))]
  yub = base[min(which(base>respy))]
  
  outside_arena = sqrt(xlb^2+ylb^2)>r & sqrt(xub^2+yub^2)>r & sqrt(xub^2+ylb^2)>r & sqrt(xlb^2+yub^2)>r
  if (outside_arena){
    p = NA
  } else{
    p = 1
  }
  return(p)
}

gaussian_prob_univariate_grid_proxy = function(r,respx,respy,truex,truey,granularity=265*2,sd = 0.1){
  #r - radius of response arena
  base = seq(from = -r,to = r,length.out = granularity)
  xlb = base[max(which(base<=respx))]
  xub = base[min(which(base>respx))]
  ylb = base[max(which(base<=respy))]
  yub = base[min(which(base>respy))]
  
  outside_arena = sqrt(xlb^2+ylb^2)>r & sqrt(xub^2+yub^2)>r & sqrt(xub^2+ylb^2)>r & sqrt(xlb^2+yub^2)>r
  if (outside_arena){
    p_x = NA
    p_y = NA
  } else{
    p_x = integrate(function(x){dnorm(x, truex ,sd = sd)}, xlb,xub)$value
    p_y = integrate(function(y){dnorm(y, truey ,sd = sd)}, ylb,yub)$value
  }
  p = data.frame(pnorm_x = p_x,pnorm_y = p_y)
  return(p)
}
