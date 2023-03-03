# FungiCLEF-22

Scripts, figures and working notes for the participation in [FungiCLEF 2022](https://www.imageclef.org/FungiCLEF2022), part of the [LifeCLEF labs](https://www.imageclef.org/LifeCLEF2022) at the [13th CLEF Conference, 2022](https://clef2022.clef-initiative.eu/index.php).

## Quick Links

The following references will help in reproducing this implementation and to extend the experiments for further analyses.

- [Manuscript [PDF]](https://ceur-ws.org/Vol-3180/paper-162.pdf)
- [Model Training Scripts](./Scripts/train)
- [Concept and Dataset Description](https://www.imageclef.org/FungiCLEF2022)

## Cite Us

[Link to the Research Paper (preprint)](https://ceur-ws.org/Vol-3180/paper-162.pdf)

If you find our work useful in your research, don't forget to cite us:

```
@article{desingu2022classification,
  url = {https://ceur-ws.org/Vol-3180/paper-162.pdf},
  title={Classification of fungi species: A deep learning based image feature extraction and gradient boosting ensemble approach},
  author={Desingu, Karthik and Bhaskar, Anirudh and Palaniappan, Mirunalini and Chodisetty, Eeswara Anvesh and Bharathi, Haricharan},
  keywords={Ensemble Learning, Convolutional Neural Networks, Gradient Boosting Ensemble, Metadata-aided Classification, Image Classification, Transfer Learning},
  journal={Conference and Labs of the Evaluation Forum},
  publisher={Conference and Labs of the Evaluation Forum},
  year={2022},
  ISSN={1613-0073},  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Key Highlights

### Proposed Prediction Workflow

- Each *observation* in the dataset is made up of numerous fungus photos and its contextual geographic information like nation, exact area where the photograph was taken on four layers, along with specific attributes like substrate and habitat. 
- Each image in an observation is preprocessed before being fed through the two feature extraction networks to generate two 4096-element-long representation vectors. - These vectors are combined with numeric encoded nation, location at three-level precision, substrate, and habitat metadata for the image to produce a final vector with a size of 8198. 
- The boosting ensemble classifier is fed all the 8198 features to generate a probability distribution over all potential fungi species classes. 

This workflow is depicted below,   
<img alt="proposed-prediction-workflow" src="./assets/Figure-1_Prediction-Workflow.png" />

### Managing Out-of-Scope Classes

- Some fungi classes in the dataset were exclusive to the test set, and were not exposed to the architicture during model training. These classes are herein referred to as *out-of-scope* classes.
- A *prediction confidence thresholding* method was devised to handle out-of-the-scope classes. 
- A threshold value is arrived at for each trained ensemble model by adopting a qualitative, trial-based method. 
- First, a histogram of maximum prediction probabilities of the model for each observation in the test set is plotted. 

The histogram for the best performing model instance is depeicted below,     
<img alt="best-threshold-histogram" src="./assets/Figure-3_Lowest-Confidence.png" />
