rm(list=ls())
library(dplyr)
library(readxl)
library(data.table)
fs = fread("all.csv",data.table=F)
data = read_xlsx("F:/ANANLYSIS/INTERACTION/csv/core.xlsx")
lc = df %>%
  select(Image,Name,celltype,Parent) %>%
  dplyr::left_join(data,by = c('Image' = 'location'))

m = {}
for (n  in unique(lc$Image)) {
  lc1 = lc %>%
    dplyr::filter(Image == x)
  all_cell = table(lc1$Image)
  for (i in unique(lc1$celltype)) {
    lc2 = lc1 %>%
      dplyr::filter(celltype == i)
    celltype = table(lc2$Image)
    m[[n]][i] = celltype/all_cell
  }
}
g = m %>% 
  do.call(rbind,.) %>%
  as.data.frame()

S = subset(g,sORm == "S")
M = subset(g,sORm == "M")
Neg = subset(g,sORm == "Neg")

S_cor  = cor(S,method = 'spearman')
M_cor  = cor(M,method = 'spearman')
Neg_cor  = cor(Neg,method = 'spearman')