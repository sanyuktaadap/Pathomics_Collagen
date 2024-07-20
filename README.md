# Collagen Biomarker

---


## Authors
Arpit Aggarwal, Himanshu Maurya, Haojia Li, and Anant Madabhushi<br>


## Packages Required
Python was used for this study.<br>
The packages required for running this code are PyTorch, Numpy, Openslide, PIL, OpenCV, Pandas, Sklearn, and Matplotlib.<br>


## Workflow for the study

<img src="example/workflow.png" width="600" height="400">


## Pipeline for Collagen biomarker
The main steps involved in the Collagen biomarker are as follows:
1. Preprocessing steps (Extracting tiles from H&E-stained Whole Slide Images and Epithelium/Stroma segmentation)
2. Extracting collagen features


## Visualization of collagen fibers
For visualizing collagen fibers for some patches before feature extraction, please transfer some patches and epi/stroma masks to 'data/patches_for_visualization' and 'data/masks_for_visualization' directories respectively. Then run the python file 'python3 code/main_visualize_collagen_fibers.py' for saving example collagen fiber masks in the results directory 'results/collagen_fiber_masks'


## Plotting heatmaps
For plotting heatmaps, please transfer some patches and epi/stroma masks to 'data/patches_for_visualization' and 'data/masks_for_visualization' directories respectively. Then run the python file 'python3 code/main_visualize_heatmaps.py' for saving example heatmaps in the results directory 'results/heatmaps_stroma' and 'results/heatmaps_peritumoral'


## Running the code
1. <b>Preprocessing steps</b><br>
a. <b>Extracting tiles from H&E-stained Whole Slide Images</b> - This extracts patches from the whole slide image of size 3000x3000-pixel. Run the python file 'python3 code/extract_patches.py' (specify the 'input_path' to the location where whole slide images exist and 'output_path' where you want to store the patches (keep it <b>'data/patches'</b>)).<br>

    b. <b>Epithelium/Stroma segmentation</b> - To segment the epithelium/stromal regions on the patches extracted above, run the pretrained epithelium/stroma model 'python3 code/epithelium_stroma_segmentation.py'. The model weights file are located at 'code/epi_seg_unet.pth' (specify the 'input_path' to the location where patches are extracted and 'output_path' where you want to store the epithelium/stroma segmentation masks (keep it <b>'data/masks'</b>)).<br>
In the epithelium/stroma mask, the white regions correspond to the tumor/epithelium areas and the black regions correspond to the stromal areas. We will use the black regions for segmenting collagen fibers and the following code automatically handles it (255-epi_stroma_mask).


2. <b>Extracting collagen features</b><br>
As described in the manuscript, for extracting the collagen features run the file (code/main_patchLevelFeatures.py) that generates the Collagen Fiber Orientation Disorder map for each patch extracted. The feature maps will be stored over here 'results/patches'.
<br><br>
After obtaining the feature maps for each tile, run the file (code/main_patientLevelFeatures.py) that gives patient-level features (mean and maximum) for each patient, giving a total of 36 features. The features for each patient will be stored over here 'results/features' which can be used for downstream tasks like survival analysis etc.<br><br>


## Survival analysis
Using the extracted features, use the notebook 'survival_analysis.ipynb' for an example demo for running the survival analysis pipeline.


## License and Usage
Madabhushi Lab - This code is made available under Apache 2.0 with Commons Clause License and is available for non-commercial academic purposes.