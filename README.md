# ColonyMap: A spatial cell colony analysis algorithm



<br><br>

<img src="./img/Fig 1.jpg" alt="Fig 1" style="zoom:100%;" />



<center>Fig1 : Schematic diagram of ColonyMap algorithm</center>

<br>
<br>


This repository includes  codes and data example used in the 'Integrative Spatial Analysis Reveals Tumor Heterogeneity and Immune Colony Niche Related to Clinical-outcomes in Small Cell Lung Cancer' paper.

<br>

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

- Code:  You can scan this script for more details  [Cell-cell_Pairwise_Interaction_Analysis.R](script/Cell-cell_Pairwise_Interaction_Analysis.R) 

  
  
  <br>
  
  <br>

## Cellular Neighborhood ldentification and Voronoi Diagram Generation

- Software：[python 3.11.5]()、[scikit-learn 1.3.0]([Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa *et al.*, JMLR 12, pp. 2825-2830, 2011.)

- Principle: Obtain a window consisting of 20 nearest neighboring cells using the Euclidean distance between X/Y coordinates, and then cluster the window using Python's scikit learn MiniBatchKMeans (k=20) function based on the composition of cell types. Then determine the cell neighborhood where each cell is located.

- Reference: [Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front.](Schürch, C. M., Bhate, S. S., Barlow, G. L., Phillips, D. J., Noti, L., Zlobec, I., Chu, P., Black, S., Demeter, J., McIlwain, D. R., Kinoshita, S., Samusik, N., Goltsev, Y., & Nolan, G. P. (2020). Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front. *Cell*, *182*(5), 1341–1359.e19. https://doi.org/10.1016/j.cell.2020.07.005)

- Code:  You can scan this script for more details  [Cellular_Neighborhood_ldentification.py](script/Cellular_Neighborhood_ldentification.py) 

  <br>
  
  <br>

## Statistical Analysis

- Software: [R 4.2.3](R Core Team (2023). _R: A Language and Environment for
  Statistical Computing_. R Foundation for Statistical
  Computing, Vienna, Austria. <https://www.R-project.org/>.)

- Cell proportion correlation analysis

  - Methods: Calculate the proportion of each cell type in each core separately, and then calculate the Spearman correlation coefficient according to the type of core.
  - Code:  You can scan this script for more details  [Statistical_Analysis.R](script/Statistical_Analysis.R) 
  
  <br>


# Data

All data have been uploaded to the GSA (https://ngdc.cncb.ac.cn/gsa/) and Zenodo (https://zenodo.org/) and will be available for download upon acceptance for publication.



<br>

# Citation

**Chen, Haiquan et al. “Integrative spatial analysis reveals tumor heterogeneity and immune colony niche related to clinical outcomes in small cell lung cancer.” Cancer cell, S1535-6108(25)00030-3. 14 Feb. 2025, doi:10.1016/j.ccell.2025.01.012**

