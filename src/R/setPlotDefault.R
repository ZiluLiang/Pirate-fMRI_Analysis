theme_set(theme_classic())
theme_update(aspect.ratio = 1,
             text = element_text(family = "sans", face = "bold"),
             plot.title = element_text(size = 16,hjust = 0.5),
             plot.subtitle = element_text(size = 14,hjust = 0.5),
             plot.margin = margin(0,0,0,0),
             axis.text= element_text(size = 12),
             axis.title = element_text(size = 14),
             legend.title = element_text(size = 12,face = "bold"),
             legend.title.align = 0.5,
             legend.text = element_text(size = 12),
             legend.text.align = 0.5,
             strip.text = element_text(size = 12,face = "bold"))
options(scipen=10)
cond_colors = ggsci::pal_aaas()(3)
twolevel_colors = c('#1f77b4', '#ff7f0e')
fourlevel_colors = ggsci::pal_aaas()(6)[c(1,2,3,4)]
y_colors = c("#00FFC1","#9E4F46","#8484FF","#005800","#FFD300") # Yellow,Quagmire,Blue,Wine,Jade
x_shapes = c("\u2BC1","\U2605","\u25CF","\u25B2","\u25A0")# diamond, star, circle, triangle, square


#fourlevel_colors = RColorBrewer::brewer.pal(11, "RdYlGn")[c(10,9,3,1)]
#loc.trialtype_colors = RColorBrewer::brewer.pal(11, "RdYlGn")[c(10,9,8,3)]
#similarity_bin_color = RColorBrewer::brewer.pal(9, "YlOrRd")[c(8,1)]
options("jtools-digits" = 3)
