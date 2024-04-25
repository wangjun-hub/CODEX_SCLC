
rm(list = ls())
load("./outres.list.rda")
library(ggplot2)
library(patchwork)

cnlist = paste0("CN", 1:20)
names(cnlist) = cnlist
p.list = lapply(cnlist, function(i){
  dat = outres.list[[i]]
  dat$gene = factor(dat$gene, levels = dat$gene)
  dat = dat[1:10,]
  dat$accuracy = 1/dat$SE
  dat$pval = ifelse(dat$P < 0.05, "sig", "non.sig")
  p1 = ggplot(dat, aes(coef,gene))+
    geom_point(aes(size=accuracy,fill=type, colour = pval),shape = 21)+
    scale_fill_manual(values = c("Gain" = "#D95745", "Loss" = "#5A8FC3",
                                 "SNV"= "#019A74")
    )+
    scale_color_manual(values = c("sig" = "black", "non.sig"= "white")) +
    scale_y_discrete(position = "right") +
    scale_x_continuous(position = "top",limits = c(0,3),breaks = c(0,3),
                       expand = c(0,0))+
    labs(y = element_blank(), x = "Coefficient",title = i) + 
    #theme_bw() +
    theme(panel.grid = element_blank(),
          panel.background = element_blank(),
          axis.text.x = element_text(colour = "black"),
          axis.text.y = element_text(colour = "black",size = 10),
          axis.line.x.top = element_line(),
          plot.title = element_text(hjust = 0.5,size = 15),
          #panel.border = element_rect(fill = NA,color="grey", linetype="solid"),
          legend.position = "none")
  
  
  p2 = ggplot(dat, aes(prop,gene))+geom_bar(aes(fill=type),stat = "identity")+
    scale_fill_manual(values = c("Gain" = "#D95745", "Loss" = "#5A8FC3",
                                 "SNV"= "#019A74")
    )+
    scale_y_discrete(position = "left") + 
    scale_x_continuous(limits = c(0,0.7),breaks = c(0,0.7),
                       expand = c(0,0))+
    labs(y = element_blank())+
    theme(panel.grid = element_blank(),
          axis.text.y = element_blank(),
          axis.title.x = element_blank(),
          axis.line.x.bottom = element_line(),
          axis.ticks.y.right = element_blank()
    )
  
  p = p1 + p2
  return(p)
}
)
p.merge = (p.list$CN1|p.list$CN2|p.list$CN3|p.list$CN4|p.list$CN5|plot_layout(nrow = 1))/
  (p.list$CN6|p.list$CN7|p.list$CN8|p.list$CN9|p.list$CN10|plot_layout(nrow = 1))/
  (p.list$CN11|p.list$CN12|p.list$CN13|p.list$CN14|p.list$CN15|plot_layout(nrow = 1))/
  (p.list$CN16|p.list$CN17|p.list$CN18|p.list$CN19|p.list$CN20|plot_layout(nrow = 1))+
  guide_area()+plot_layout(guides = "collect")
ggsave(p.merge, file = "p.merge.pdf", width = 16,height = 12)

