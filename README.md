
# Enhanced Leaf Disease Detection:UNet for segmentation and optimized EfficientNet for classification

A hybrid approach based on optimal automatic DL is presented in this study for the classification of plant leaf diseases (PLDC). First, pre-processing is carried out using the suggested model utilizing Gaussian filtering and image scaling. Subsequently, the UNet technique is employed to segment the disease-infected zone in order to obtain the relevant region and improve the accuracy of illness categorization. The hunter-prey optimization (Hunt-PO) algorithm was used to fine-tune the UNet model's weight during segmentation. After that, a Gabor filter, scale-invariant feature transform (SIFT), and gray level co-occurrence matrix (GLCM) are used in feature extraction to obtain the important characteristics for classification. Additionally, artificial driving-EfficientNet (AD-ENet) is used to execute PLDC based on the retrieved features. 

## Work Published in following article
Kotwal, J., Kashyap, R., Pathan, S., 2023. Artificial Driven based EfficientNet for Automatic Plant Leaf Disease Classification, Multimedia Tool’s Application, https://doi.org/10.1007/s11042-023-16882-w.(Link:https://link.springer.com/article/10.1007/s11042-023-16882-w)

Kotwal, J., Kashyap, R., Pathan, S., 2023. An India soyabean dataset for identification and classification of diseases using computer-vision algorithms, Data in Brief, https://doi.org/10.1016/j.dib.2024.110216.(Link:https://www.sciencedirect.com/science/article/pii/S2352340924001872)

### Folder
Folder consists of 
 1) Hybrid Humming Bird optimization
 2) Dataset folder contains input image, filtered image and segmented image.
 3) Model folder contains the existing model.
 4) Results folder conatins all the images.
 5) Output images folder contains the input image, filtered image and segmented image. 

 ## Dataset
1) PlantVillage Dataset
2) An Indian soyabean dataset(Own dataset collected from Maharashtra region).

##Model
1) EfficientNet Model
    - Model.py file contains the Efficient model
2) UNet Model
    - UNet.py file used to segment the image to get the exact location segmented where the diseases is affected.
    - Segmented and mask image is shown in Output folder.

## Video Link of Model
      Enhanced Leaf Disease Detection:UNet for segmentation and optimized EfficientNet for classification:(Link- https://drive.google.com/file/d/16kEj7MwrLafxBz_SKRrLlCGn7bF1D-do/view?usp=sharing)

## References
1) Kotwal, J., Kashyap, R., Pathan, S., 2024. Yolov5-based convolutional Feature Attention Neural Network for Plant Disease Classification, international Journal of Intelligent Systems Technologies and Applications, https://doi.org/10.1504/IJISTA.2024.10062157.
2) Kotwal, J., Kashyap, R., Pathan, S., 2023. Agricultural Plant diseases identification: from traditional approach to deep learning, Material Today: Proceeding, https://doi.org/10.1016/j.matpr.2023.02.370.
3) Kotwal, J., Kashyap, R., Pathan, S., 2023. Artificial Driven based EfficientNet for Automatic Plant Leaf Disease Classification, Multimedia Tool’s Application, https://doi.org/10.1007/s11042-023-16882-w.
4) Kotwal, J., Kashyap, R., Pathan, S., 2023. An India soyabean dataset for identification and classification of diseases using computer-vision algorithms, Data in Brief, https://doi.org/10.1016/j.dib.2024.110216.
