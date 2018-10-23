# Classifier implementations for Deep Neural Network and Clustered Support Vector Machine
Implementations for two classification algorithms for a two-dimensional multi-class dataset. This is a stand-alone project created to compare with our piecewise linear classification model which is avaialable at <a href="https://github.com/abhiachoudhary/Piecewise-Linear-Model-For-Nonconvex-Classifiers">this link</a>.

## PYTHON files description
`DNN_Classifier.py`: This is the main implementation script for DNN using TensorFlow. The `get_data` routine lets you choose among different available datasets from Data folder or create a new dataset by calling `create_data.py` script. The remaining script is standard neural network functions with customizable choice of parameters (layers, activation function, training rate, gradient scheme etc.). The output is accuracy along with a pictorial view of classifier.

`Gu_Han.py`: This is an implementation of the classification algorithm by `[Gu and Han, 2013]`. The technique is based on dividing entire dataset into multiple clusters and applying localized support vector machine within each cluster. The output is cumulative accuracy as well as accuracy within each cluster along with a pictorial view of all classifiers.

`create_data.py`: This script creates various types of two-dimensional datasets that are classifiable by different functions prescribed through a list of parameters. 

`plot_boundary.py`: This script plots the data points for both training and test sets along with the generated decision boundary from either of the classification algorithms. 

## List of data files included
* `2D_LSD.csv`: Linearly separable data
* `2D_PLSD.csv`: Piecewise linearly separable data (2 lines)
* `2D_PPLSD.csv`: Piecewise linearly separable data (3 lines)
* `2D_polySD.csv`: Polynomially separable data
* `2D_circlesSD.csv`: Data separable by a circular decision boundary 


## References
Gu, Q., Han, J.: Clustered support vector machines. In: Artificial Intelligence and Statistics, pp. 307-315 (2013)

## Feedback
Email your feedback to <a href="mailto:abhi.achoudhary@gmail.com">Abhishek Choudhary</a>.
