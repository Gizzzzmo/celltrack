# Cell image segmentation with the help of differentiable rendering

This project aims to segment cell images of the GFP-GOWT1 dataset from the [cell tracking challenge](http://celltrackingchallenge.net/2d-datasets/).

The module *blobs* models cells as ellipses, more specifically as gaussian functions, parameterized by their center, a matrix and a brightness. 

The module *stage1* models cells as polygons, more specifically as triangle meshes, that are rendered with the help of the pyredner raytracer. The results from the *blob* module can optionally be used as a starting point of this, more fine grained, optimization.

The module *stage2* takes the produced vertex lists from *stage1*, and creates a dataset of cutouts from the original images. With the help of the ground truth segmentation masks these cutouts are either labeled as containing or not containing an actual cell.

The module *train* has a number of functions to train different classifiers on the dataset generated by *stage2*.

The module *validate*, with the help of *train*, first trains different classifiers on part of the data, then evaluates their performance on a separate dataset.

The module *setup* defines various hyperparameters, that are used throughout different modules. It also contains miscellaneous helper functions. 

The other modules contain helper functions, for dealing with cells, setting up a scene with pyredner, loading  and visualizing data, etc.

Requirements are 
- pytorch
- numpy
- matplotlib
- sortedcontainers
- scikit-image
- pyredner and redner
