# ColonyMAP: A spatial cell colony analysis algorithm





<img src="./img/Fig 1.jpg" alt="Fig 1" style="zoom:100%;" />



<center>Fig1 : Macrophage colonies (left) & SCLA-A cancer cell colonies (right)</center>



This repository includes  codes and data example **(Colony Part)** used in the "Spatial evolution and colony landscape of small cell lung cancer" paper.





[TOC]

## Summary📖

Structured cell colonies (CC) and dispersed cells within tissues exhibit distinctly different physiological functions. In situ CCs, coordinated with homogeneous cells, have proven to be invaluable for histopathological detection and diagnosis. Cell neighborhood method clustering cell composition to cell pattern (CN) by traversal small regional tissue. In contrast, the colony perspective excels at pinpointing intercellular boundary and interactions regions. To analyze cell colonies in CODEX data, we developed ColonyMap, a spatial cell colony analysis algorithm.





## Requirements🌸

●**Python (version 3.8.17)**

- [ ] opencv package(version 4.5.1.48)

- [ ] numpy package(version 1.24.3)

- [ ] matplotlib package(version 3.7.1)

- [ ] pickle package(version 4.0)

●**R (version version 4.3.1)**

- [ ] spatstat (version 3.0_6)



☞ More details in **requirements.txt**





## Usage👻

To run this analysis pipline, you need to first create a python virtual environment (Python : 3.8.17 & R : 4.3.1) where you can install all the required packages. If you are using the conda platform to create your virtual environment. These are the steps you need to follow.



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



●`Colony_show.py`  is used to display the spatial distribution of  cellt colonies , such as macrophage colonies.

```
cd [your_path]/script
```

and then

```
python Colony_macrophage_show.py "../data"
```

It takes only 1 second to visualize the macrophage colonies on an image. (Computing machine: Macbook Air  M1 core)



<img src="./img/Fig 2.jpg" alt="Fig 2" style="zoom:100%;" />

- <center>Fig2 : Macrophage colonies (left) & SCLA-A cancer cell colonies (right)</center>

  



●`Subburst_Chart.R`  is used to visualize the strength of interactions between different cell colonies.

```
Rscript Subburst_Chart.R 
```





<img src="./img/Fig 3.jpg" alt="Fig 3" style="zoom:100%;" />

<center>Fig3 : The interaction intensity in different immune celltypes</center>



☞  You can scan this file for dynamic click interaction  [Subburst_Chart.html](result/Subburst_Chart.html) 








## Data

To be supplemented





## Citation

To be supplemented

