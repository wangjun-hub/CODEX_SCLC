rm(list = ls())
options(stringsAsFactors = F)
library(ggplot2)
library(imcRtools)
library(cytomapper)
library(RColorBrewer)
library(ComplexHeatmap)
library(dplyr)
library(tidyr)
library(Seurat)
library(data.table)

cur_features = fread("all.csv",data.table=F)
counts <- cur_features[,grepl("Cell: Mean", 
                                  colnames(cur_features))]
meta <- cur_features[,c(1:5,8:21,762)]
coords <- cur_features[,c("Centroid X µm", "Centroid Y µm")]
colnames(coords) = c("Pos_X", "Pos_Y")
cur_ch <- strsplit(colnames(counts), ":")
colnames(counts) = sapply (cur_ch,function(x){x[1]}) 
spe3 <- SpatialExperiment(assays = list(counts = t(counts)),
                          colData = meta, 
                          sample_id = as.character(meta$Image),
                          spatialCoords = as.matrix(coords))
colnames(spe3) <- paste0(spe3$sample_id, "_", 1:length(counts))
assay(spe3, "exprs") <- asinh(counts(spe3)/1)
rowData(spe3)$use_channel <- !grepl("DAPI", rownames(spe3))
spe3 <- buildSpatialGraph(spe3, img_id = "Image", type = "knn", k = 20)
out2 <- testInteractions(spe3,
                        group_by = "Image",
                        label = "celltype",
                        method = "classic",
                        colPairName = "knn_interaction_graph",
                        BPPARAM = BiocParallel::SerialParam(RNGseed = 123))