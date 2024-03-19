library(plotly)

data_plot<-read.table(file="../data/Subburst_Chart_data.csv",header = T,sep = ",")


p1<-plot_ly(data = data_plot,color = I("black"),size = I(30),
            labels = ~label, 
            parents = ~parent,
            values = ~value,
            type='sunburst'
            #color = data_plot$label
)

p1

pal<-c("#7E6148FF","#00A087FF","#DC0000FF","#E64B35FF","#F39B7FFF","#4DBBD5FF","#91D1C2FF","#3C5488FF","#8491B4FF","#B09C85FF")
p2<-p1%>%layout(colorway=pal)

p2


