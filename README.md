# Collagen Biomarker

---


## Authors
Arpit Aggarwal, Himanshu Maurya, and Anant Madabhushi<br>


## Packages Required
Python was used for this study.<br>
The packages required for running this code are PyTorch, Numpy, Openslide, PIL, OpenCV, Pandas, Sklearn, and Matplotlib.<br>


## Workflow for the study

<img src="example/workflow.png" width="800" height="500">


## Pipeline for Collagen biomarker
The main steps involved in the Collagen biomarker are as follows:
1. Preprocessing steps (Extracting tiles from H&E-stained Whole Slide Images and Epithelium/Stroma segmentation)
2. Extracting collagen features


## Running the code
1. <b>Preprocessing steps</b><br>
a. <b>Extracting tiles from H&E-stained Whole Slide Images</b> - This extracts tiles from the whole slide image of size 3000x3000-pixel. Run the python file 'python3 code/extract_patches.py' (specify the 'input_path' to the location where whole slide images exist and 'output_path' where you want to store the tiles).<br>


b. <b>Epithelium/Stroma segmentation</b> - To segment the epithelium/stromal regions on the tiles extracted above, run the pretrained epithelium/stroma model 'python3 code/epithelium_stroma_segmentation.py'. The model weights file is located at 'code/epi_seg_unet.pth' (specify the 'input_path' to the location where tiles are extracted and 'output_path' where you want to store the epithelium/stroma segmentation masks).<br>


2. <b>Extracting collagen features</b><br>
As described in the manuscript, for extracting the collagen features run the file (code/main_patchLevelFeatures.py) that generates the Collagen Fiber Orientation Disorder map for each tile extracted.
<br><br>
After obtaining the feature maps for each tile, run the file (code/main_patientLevelFeatures.py) that gives patient-level features (mean and maximum) for each patient, giving a total of 36 features. <br><br>