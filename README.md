# GEM-balls

## What is this

GEM-balls is a powerful machine learning algorithm for binary classification. 

You can find all the information about it in the original paper:

["A New Classification Algorithm With Guaranteed Sensitivity and Specificity for Medical Applications",
  by A. Carè, F.A. Ramponi, M.C. Campi.  IEEE Control Systems Letters, vol. 2, no. 3, pp. 393-398, July 2018.](http://www.algocare.it/L-CSL2018GEM.pdf)


## How to install

You can install the package using pip:

    pip install gemballs
    
    
## Simple usage
  
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_circles
    from gemballs.gemballspollingclassifier import GEMBallsPollingClassifier

    # make some artificial data
    X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=666)

    # split data in train and test
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=7)

    gemballs = GEMBallsPollingClassifier()

    # fit the model
    gemballs.fit(X_train, y_train)

    # now the model is fitted, we can predict value

    # show the score
    print(gemballs.score(X_test, y_test))

## Plots

![Plot](/plot.png)
