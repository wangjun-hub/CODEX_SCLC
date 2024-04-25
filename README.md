# ColonyMAP: A spatial cell colony analysis algorithm



<br><br>

<img src="./img/Fig 1.jpg" alt="Fig 1" style="zoom:100%;" />



<center>Fig1 : Schematic diagram of ColonyMAP algorithm</center>

<br>

This repository includes  codes and data example used in the 'Spatial evolution and colony landscape of small cell lung cancer' paper.

<br>

[TOC]



<br><br>

<br>

## Summary📖

Structured cell colonies (CC) and dispersed cells within tissues exhibit distinctly different physiological functions. In situ CCs, coordinated with homogeneous cells, have proven to be invaluable for histopathological detection and diagnosis. Cell neighborhood method clustering cell composition to cell pattern (CN) by traversal small regional tissue. In contrast, the colony perspective excels at pinpointing intercellular boundary and interactions regions. To analyze cell colonies in CODEX data, we developed ColonyMap, a spatial cell colony analysis algorithm.

<br><br>

<br>

## Requirements🌸

●**Python (version 3.8.17)**

- [ ] opencv package(version 4.5.1.48)

- [ ] numpy package(version 1.24.3)

- [ ] matplotlib package(version 3.7.1)

- [ ] pickle package(version 4.0)

●**R (version version 4.3.1)**

- [ ] spatstat (version 3.0_6)

<br><br>

☞ More details in **requirements.txt**



<br><br>

<br>

## Usage👻

To run this analysis pipline, you need to first create a python virtual environment (Python : 3.8.17 & R : 4.3.1) where you can install all the required packages. If you are using the conda platform to create your virtual environment. These are the steps you need to follow.

<br>

#### Virtual environment construction

All operations are completed in your terminal.

First, let us create the virtual environment with all packages: 

```
conda create -n [virtual_environment_name] --file requirements.txt
```

Next, you need to activate the virtual environment:

```
conda activate [virtual_environment_name]
```

Once you have the `[virtual_environment_name]` ready, you can run all scripts!

<br>

#### Script Description

●`Colony_recognition.py` is used to identify the colony contours of all cell types in the image, which is the cornerstone of subsequent analysis. So you need to run it first.

```
cd [your_path]/script
```

and then

```
python Colony_recognition.py "../data"
```

It takes approximately 20 seconds to identify the contours of all cell types on an image of approximately 40000 cells. (Computing machine: Macbook Air  M1 core)

<br><br>

●`Colony_show.py`  is used to display the spatial distribution of  cellt colonies , such as macrophage colonies.

```
cd [your_path]/script
```

and then

```
python Colony_macrophage_show.py "../data"
```

It takes only 1 second to visualize the macrophage colonies on an image. (Computing machine: Macbook Air  M1 core)

<br>



<img src="./img/Fig 2.jpg" alt="Fig 2" style="zoom:100%;" />



<center>Fig2 : Macrophage colonies (left) & SCLA-A cancer cell colonies (right)</center>

<br><br>



●`Subburst_Chart.R`  is used to visualize the strength of interactions between different cell colonies.

```
Rscript Subburst_Chart.R 
```

<br>



<img src="./img/Fig 3.jpg" alt="Fig 3" style="zoom:100%;" />

<center>Fig3 : The interaction intensity in different immune celltypes</center>

 <br><br>

☞  You can scan this file for dynamic click interaction  [Subburst_Chart.html](result/Subburst_Chart.html) 







 <br><br>

 <br><br>

# Supplementary analysis pipeline 

## Computational lmage Processing

- Software：[qupath v0.3.2](Bankhead, P. et al. QuPath: Open source software for digital pathology image analysis. Scientific Reports (2017),https://doi.org/10.1038/s41598-017-17204-5.)

- Methods：

  - Before initiating data analysis, quality control was performed on each individual image by visual assessment across the whole slide; each marker was qualitatively evaluated based on signal intensity compared with the background and for staining specificity. Other artifacts such as out-of-focus regions, tissue folding, and debris were manually annotated and excluded from the analysis. 

  - Organize the segmentation using[Pixel classification — QuPath 0.5.1 documentation](https://qupath.readthedocs.io/en/stable/docs/tutorials/pixel_classification.html). First, multiple small regions belonging to the tumor, mesenchymal, and necrotic categories are annotated, and the correct tissue classification is judged by experts. If there is an error in the result, add a category annotation to improve the accuracy until the HE result is met.

    <br>

    <br>

## Single-cell Segmentation

- Software：[qupath v0.3.2](Bankhead, P. et al. QuPath: Open source software for digital pathology image analysis. Scientific Reports (2017),https://doi.org/10.1038/s41598-017-17204-5.),[StarDist v0.3.2](Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell Detection with Star-convex Polygons. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.)

- Methods：Nuclear segmentation was first performed using the deep learning method StarDist, utilizing its default parameters, and applying the [2D_dsb2018 model to the DAPI image](Schmidt U, Weigert M, Broaddus C, et al. Cell detection with star-convex polygons. In: Medical Image Computing and Computer AssistedIntervention—MICCAI 2018. (Frangi AF, Schnabel JA, Davatzikos C, et al.eds.) Lecture Notes in Computer Science Springer International Publishing:Cham; 2018; pp. 265–273; doi: 10.1007/978-3-030-00934-2_30).To align with the training set’s image resolution and enhance image contrast for prediction, preprocessing steps were implemented, which included reducing the image size by 50% and applying Contrast Limited Adaptive Histogram Equalization.The cytoplasmic segmentation was established by expanding the nuclei through a morphological dilation of 5 μm, which was applied to the labeled nuclear mask. The centroid of each cell was subsequently determined using the x-y coordinates of the nuclear object’s centroid within the image. A comprehensive qualitative assessment of the segmentation was carried out for each individual slide, and it yielded consistently satisfactory results, affirming the robustness and reliability of the segmentation process.

  <br>

  <br>

## Single-cell Lineage Assignment

- Software：[qupath v0.3.2](Bankhead, P. et al. QuPath: Open source software for digital pathology image analysis. Scientific Reports (2017),https://doi.org/10.1038/s41598-017-17204-5.)

- Methods: After cell segmentation, calculate the average fluorescence intensity of each marker in each cell based on the mask segmentation mask and fluorescence image. Reference [qupath machine learning process]([Multiplexed analysis — QuPath 0.5.1 documentation](https://qupath.readthedocs.io/en/stable/docs/tutorials/multiplex_analysis.html#create-training-images)). Based on the co expression and hierarchical relationship between markers, multiple classifiers are established, including `PanCK-CD68-CD3e-CD20`、`ASCL1`、`NEUROD1`、`POU2F3`、`YAP1`、`CD4-CD8-CD15-CD31`、`CD56`、`CD11c`、`Foxp3` . After the model is applied, it is confirmed by experts for a second time. If there are false positives or false negatives, the accuracy of cell typing can be improved by increasing the training dataset and setting a threshold (gating) method. For functional markers, set thresholds to determine their expression based on their expression status.

  <br>

  <br>

## Cell-cell Pairwise Interaction Analysis

- Software: [imcRtools 1.8.0](Windhager, J., Zanotelli, V.R.T., Schulz, D. et al. An end-to-end workflow for multiplexed image processing and analysis. Nat Protoc (2023). https://doi.org/10.1038/s41596-023-00881-0)  

- Principle: Define each cell and its surrounding neighbors through the Euclidean distance between X/Y coordinates. The increase or decrease in interactions between cell types is compared using random tissues matched with each image, and the permutation test method of Monte Carlo sampling is used to test whether interactions are significantly enriched or decreased.

- Code

  ```R
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
  ```

  <br>

  <br>

## Cellular Neighborhood ldentification and Voronoi Diagram Generation

- Software：[python 3.11.5]()、[scikit-learn 1.3.0]([Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa *et al.*, JMLR 12, pp. 2825-2830, 2011.)

- Principle: Obtain a window consisting of 20 nearest neighboring cells using the Euclidean distance between X/Y coordinates, and then cluster the window using Python's scikit learn MiniBatchKMeans (k=20) function based on the composition of cell types. Then determine the cell neighborhood where each cell is located.

- Reference: [Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front.](Schürch, C. M., Bhate, S. S., Barlow, G. L., Phillips, D. J., Noti, L., Zlobec, I., Chu, P., Black, S., Demeter, J., McIlwain, D. R., Kinoshita, S., Samusik, N., Goltsev, Y., & Nolan, G. P. (2020). Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front. *Cell*, *182*(5), 1341–1359.e19. https://doi.org/10.1016/j.cell.2020.07.005)

  ```python
  import pandas as pd
  import numpy as np
  from matplotlib.colors import  LinearSegmentedColormap
  from sklearn.neighbors import NearestNeighbors
  import time
  import sys
  import matplotlib.pyplot as plt
  from sklearn.cluster import MiniBatchKMeans
  import seaborn as sns
  
  
  
  def get_windows(job,n_neighbors):
      '''
      For each region and each individual cell in dataset, return the indices of the nearest neighbors.
      
      'job:  meta data containing the start time,index of region, region name, indices of region in original dataframe
      n_neighbors:  the number of neighbors to find for each cell
      '''
      start_time,idx,tissue_name,indices = job
      job_start = time.time()
      
      print ("Starting:", str(idx+1)+'/'+str(len(exps)),': ' + exps[idx])
  
      tissue = tissue_group.get_group(tissue_name)
      to_fit = tissue.loc[indices][[X,Y]].values
  
  #     fit = NearestNeighbors(n_neighbors=n_neighbors+1).fit(tissue[[X,Y]].values)
      fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X,Y]].values)
      m = fit.kneighbors(to_fit)
  #     m = m[0][:,1:], m[1][:,1:]
      m = m[0], m[1]
      
  
      #sort_neighbors
      args = m[0].argsort(axis = 1)
      add = np.arange(m[1].shape[0])*m[1].shape[1]
      sorted_indices = m[1].flatten()[args+add[:,None]]
  
      neighbors = tissue.index.values[sorted_indices]
     
      end_time = time.time()
     
      print ("Finishing:", str(idx+1)+"/"+str(len(exps)),": "+ exps[idx],end_time-job_start,end_time-start_time)
      return neighbors.astype(np.int32)
  
  ```

  ```R
  ks = [20] # k=5 means it collects 5 nearest neighbors for each center cell
  path_to_data = 'all.csv'
  X = 'X:X'
  Y = 'Y:Y'
  reg = 'Image'
  file_type = 'csv'
  
  cluster_col = 'celltype'
  keep_cols = [X,Y,reg,cluster_col]
  save_path = ''
  ```

  ```python
  cells = pd.read_csv(path_to_data)
  cells.head()    
  #read in data and do some quick data rearrangement
  n_neighbors = max(ks)
  #assert (file_type=='csv' or file_type =='pickle') #
  
  
  #if file_type == 'pickle':
  #    cells = pd.read_pickle(path_to_data)
  #if file_type == 'csv':
   #   cells = pd.read_csv(path_to_data)
  
  cells = pd.concat([cells,pd.get_dummies(cells[cluster_col])],axis = 1)
  
  
  #cells = cells.reset_index() #Uncomment this line if you do any subsetting of dataframe such as removing dirt etc or will throw error at end of next next code block (cell 6)
  
  sum_cols = cells[cluster_col].unique()
  values = cells[sum_cols].values
  
  #find windows for each cell in each tissue region
  tissue_group = cells[[X,Y,reg]].groupby(reg)
  exps = list(cells[reg].unique())
  tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)] 
  tissues = [get_windows(job,n_neighbors) for job in tissue_chunks]
  
  #for each cell and its nearest neighbors, reshape and count the number of each cell type in those neighbors.
  out_dict = {}
  for k in ks:
      for neighbors,job in zip(tissues,tissue_chunks):
  
          chunk = np.arange(len(neighbors))#indices
          tissue_name = job[2]
          indices = job[3]
          window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(sum_cols)).sum(axis = 1)
          out_dict[(tissue_name,k)] = (window.astype(np.float16),indices)
  
  #concatenate the summed windows and combine into one dataframe for each window size tested.
  windows = {}
  for k in ks:
     
      window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = sum_cols) for exp in exps],axis = 0)
      window = window.loc[cells.index.values]
      window = pd.concat([cells[keep_cols],window],axis = 1)
      windows[k] = window
             
  k = 20
  n_neighborhoods = 20
  neighborhood_name = "neighborhood"+str(k)
  k_centroids = {}
  
  windows2 = windows[20]
  # windows2[cluster_col] = cells[cluster_col]
  
  km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
  
  labelskm = km.fit_predict(windows2[sum_cols].values)
  k_centroids[k] = km.cluster_centers_
  cells['neighborhood20'] = labelskm
  cells[neighborhood_name] = cells[neighborhood_name].astype('category')
  #['reg064_A','reg066_A','reg018_B','reg023_A']
  
  colors = [(0,'#dcdedf'),(1,'#4550cd')]
  cmp = LinearSegmentedColormap.from_list('custom_cmp',colors)
  cell_order = ["cancer cell type1","cancer cell type2",
                "cancer cell type3","cancer celltype4",
                "mix","negative","Cytotoxic T cell",
                "T helper","T reg","T other","NK cell","NK-like T",
                "B cell","Macrophage","Dendritic cell",
                "Neutrophil","Endothelial cell","Unclassified"]
  # this plot shows the types of cells (ClusterIDs) in the different niches (0-7)
  k_to_plot = 20
  niche_clusters = (k_centroids[k_to_plot])
  tissue_avgs = values.mean(axis = 0)
  fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
  fc = pd.DataFrame(fc,columns = sum_cols)
  s=sns.clustermap(fc.loc[:,cell_order], vmin =-3,vmax = 3,cmap = cmp,row_cluster = False)
  ```

- Voronoi

  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  # from shapely.ops import polygonize,unary_union
  from shapely.geometry import MultiPoint, Point, Polygon
  from scipy.spatial import Voronoi
  
  
          
  def voronoi_finite_polygons_2d(vor, radius=None):
      """
      adapted from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647 3.18.2019
      
      
      Reconstruct infinite voronoi regions in a 2D diagram to finite
      regions.
  
      Parameters
      ----------
      vor : Voronoi
          Input diagram
      radius : float, optional
          Distance to 'points at infinity'.
  
      Returns
      -------
      regions : list of tuples
          Indices of vertices in each revised Voronoi regions.
      vertices : list of tuples
          Coordinates for revised Voronoi vertices. Same as coordinates
          of input vertices, with 'points at infinity' appended to the
          end.
  
      """
  
      if vor.points.shape[1] != 2:
          raise ValueError("Requires 2D input")
  
      new_regions = []
      new_vertices = vor.vertices.tolist()
  
      center = vor.points.mean(axis=0)
      if radius is None:
          radius = vor.points.ptp().max()
  
      # Construct a map containing all ridges for a given point
      all_ridges = {}
      for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
          all_ridges.setdefault(p1, []).append((p2, v1, v2))
          all_ridges.setdefault(p2, []).append((p1, v1, v2))
  
      # Reconstruct infinite regions
      for p1, region in enumerate(vor.point_region):
          vertices = vor.regions[region]
  
          if all(v >= 0 for v in vertices):
              # finite region
              new_regions.append(vertices)
              continue
  
          # reconstruct a non-finite region
          ridges = all_ridges[p1]
          new_region = [v for v in vertices if v >= 0]
  
          for p2, v1, v2 in ridges:
              if v2 < 0:
                  v1, v2 = v2, v1
              if v1 >= 0:
                  # finite ridge: already in the region
                  continue
  
              # Compute the missing endpoint of an infinite ridge
  
              t = vor.points[p2] - vor.points[p1] # tangent
              t /= np.linalg.norm(t)
              n = np.array([-t[1], t[0]])  # normal
  
              midpoint = vor.points[[p1, p2]].mean(axis=0)
              direction = np.sign(np.dot(midpoint - center, n)) * n
              far_point = vor.vertices[v2] + direction * radius
  
              new_region.append(len(new_vertices))
              new_vertices.append(far_point.tolist())
  
          # sort region counterclockwise
          vs = np.asarray([new_vertices[v] for v in new_region])
          c = vs.mean(axis=0)
          angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
          new_region = np.array(new_region)[np.argsort(angles)]
  
          # finish
          new_regions.append(new_region.tolist())
  
      return new_regions, np.asarray(new_vertices)
  
  def plot_voronoi(points,colors,invert_y = True,edge_color = 'facecolor',line_width = .1,alpha = 1,size_max=np.inf):
      
  # spot_samp = spot#.sample#(n=100,random_state = 0)
  # points = spot_samp[['X:X','Y:Y']].values
  # colors = [sns.color_palette('bright')[i] for i in spot_samp['neighborhood10']]
  
      if invert_y:
          points[:,1] = max(points[:,1])-points[:,1]
      vor = Voronoi(points)
  
      regions, vertices = voronoi_finite_polygons_2d(vor)
  
      pts = MultiPoint([Point(i) for i in points])
      mask = pts.convex_hull
      new_vertices = []
      if type(alpha)!=list:
          alpha = [alpha]*len(points)
      areas = []
      for i,(region,alph) in enumerate(zip(regions,alpha)):
          polygon = vertices[region]
          shape = list(polygon.shape)
          shape[0] += 1
          p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
          areas+=[p.area]
          if p.area <size_max:
              poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
              new_vertices.append(poly)
              if edge_color == 'facecolor':
                  plt.fill(*zip(*poly), alpha=alph,edgecolor=  colors[i],linewidth = line_width , facecolor = colors[i])
              else:
                  plt.fill(*zip(*poly), alpha=alph,edgecolor=  edge_color,linewidth = line_width, facecolor = colors[i])
          # else:
  
          #     plt.scatter(np.mean(p.boundary.xy[0]),np.mean(p.boundary.xy[1]),c = colors[i])
      return areas
  def draw_voronoi_scatter(spot,c,voronoi_palette = sns.color_palette('bright'),scatter_palette = 'voronoi',X = 'X:X', Y = 'Y:Y',voronoi_hue = 'neighborhood10',scatter_hue = 'ClusterName',
          figsize = (5,5),
           voronoi_kwargs = {},
           scatter_kwargs = {}):
      if scatter_palette=='voronoi':
          scatter_palette = voronoi_palette
          scatter_hue = voronoi_hue
      '''
      plot voronoi of a region and overlay the location of specific cell types onto this
      
      spot:  cells that are used for voronoi diagram
      c:  cells that are plotted over voronoi
      palette:  color palette used for coloring neighborhoods
      X/Y:  column name used for X/Y locations
      hue:  column name used for neighborhood allocation
      figsize:  size of figure
      voronoi_kwargs:  arguments passed to plot_vornoi function
      scatter_kwargs:  arguments passed to plt.scatter()
  
      returns sizes of each voronoi to make it easier to pick a size_max argument if necessary
      '''
      if len(c)>0:
          neigh_alpha = .3
      else:
          neigh_alpha = 1
          
      voronoi_kwargs = {**{'alpha':neigh_alpha},**voronoi_kwargs}
      scatter_kwargs = {**{'s':50,'alpha':1,'marker':'.'},**scatter_kwargs}
      
      plt.figure(figsize = figsize)
      colors  = [voronoi_palette[i] for i in spot[voronoi_hue]]
      a = plot_voronoi(spot[[X,Y]].values,
                   colors,#[{0:'white',1:'red',2:'purple'}[i] for i in spot['color']],
                   **voronoi_kwargs)
      
      if len(c)>0:
          if 'c' not in scatter_kwargs:
              colors  = [scatter_palette[i] for i in c[scatter_hue]]
              scatter_kwargs['c'] = colors
              
          plt.scatter(x = c[X],y = (max(spot[Y])-c[Y].values),
                    **scatter_kwargs
                     )
      plt.axis('off');
      return a
  ```

  ```python
  import pandas as pd
  from voronoi import draw_voronoi_scatter
  cells2 = pd.read_csv("nh.csv")
  
  spot = cells2[cells2['File Name']=='I1']
  _ = draw_voronoi_scatter(spot,[],)
  ```

  <br>

  <br>

## Statistical Analysis

- Software: [R 4.2.3](R Core Team (2023). _R: A Language and Environment for
  Statistical Computing_. R Foundation for Statistical
  Computing, Vienna, Austria. <https://www.R-project.org/>.)

- Cell proportion correlation analysis

  - Methods: Calculate the proportion of each cell type in each core separately, and then calculate the Spearman correlation coefficient according to the type of core.

  ```R
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
  ```

  <br>


# Data

To be supplemented



<br>

# Citation

To be supplemented

