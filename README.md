# Clean_Outliers
Clean the outliers of the created dataset

# How to run a demo
```sh
$ python clean_outliers.py path/to/the/Dataset
```

# Version Control
## Version 1
- **Status**: At this version, We use the pretrained ResNet with Pytorch to classify each image in the created dataset and keep the class number of each image.
- **TODO**: Using the class numbers, we would like to find out and delete the outliers.

Classification result for cat dataset with 130 images
![image](https://github.com/Chen-Xieyuanli/clean_outliers/blob/master/classification_result_130.png)

Classification result for cat dataset with 3000 images
![image](https://github.com/Chen-Xieyuanli/clean_outliers/blob/master/classification_result_3000.png)
