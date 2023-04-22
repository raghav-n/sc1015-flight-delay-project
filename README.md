# sc1015-flight-delay-project [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/raghav-n/sc1015-flight-delay-project/main)
A project driven by machine learning to predict flight delays in the US. 

> Our team's members are Kapoor Arunaksh (U2222828L), Jodius Low (U2221134H), and Raghav Narayanswamy (U2222228G).

## Introduction
Today, travelling by air is a critical mode of transportation. One of the most frustrating experiences for any traveller is dealing with flight delays, which can impact everything from personal schedules to business appointments. We wanted to develop a model that would accurately estimate the delay in departure and arrival times of flights. The problem statement we are addressing is: **How can we predict flight delays, and help people reduce their waiting time for flights?**

For our model, we will focus on predicting arrival delays; two subquestions then emerge: (1) can we predict **whether** a flight will be delayed and (2) can we predict **how much** it will be delayed by?

## Datasets
We used two datasets to address our problem: one with the main flight data, which includes many of the key features as well as the response variable, as well as a weather dataset to bring in weather data at the departure and arrival airports. Our focus is on domestic flights in the United States for this analysis. 
1. **Flight data**, from Kaggle: *Airline Delay and Cancellation Data, 2009 - 2018* compiled by Yuanyu Mu.
2. **Weather data**, from the National Centers for Environmental Information in the US.

## Notebooks: walkthrough
The key points of the machine learning process, including data preparation, cleaning, EDA, and the final machine learning models, for both regression and classification, are detailed below.

### 1.	Preprocessing flight data ([process_flights.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/process_flights.ipynb "process_flights.ipynb"))
We apply the following steps to cut down the data involved, which is over 5 million rows if we consider all data from 2009 to 2018:
- We only consider data from 2017 and 2018.
- We filter to the top 10 busiest airports over the period 2017–2018.
- For each individual flight number, we only consider the data from every fifth day. The remaining four days prior to the fifth day are used to engineer new features called 'previous average delay' for both arrival and departure. We hypothesize this new feature will provide an indication of possible delays in the next day's flight. Note that it is important to choose a number which is not a multiple of 7 for this, so that we aren't just predicting the same day of the week each time.
- To prepare the data for classification, we treat flights which depart or arrive late by 10 minutes or more as 'delayed', and create the appropriate categorical (binary) response variable.

Note that we also remove cancelled and diverted flights as the response variable is unavailable for prediction.

### 2.	Preprocessing weather data ([process_weather.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/process_weather.ipynb "process_weather.ipynb"))
The weather data is not as clean as the flight data, and there is a lot of missing data, as well as wide variance across airports in the way in which data is reported. We apply the following steps to prepare the data for merging with the flight data and the machine learning phase.
- The airport callsigns are provided in ICAO, rather than IATA, codes. Some of them are also blank. So we use another column, `STATION`, which is never blank, to map each row to the correct codes for merging with the flight dataset, which has airports labeled with IATA representation.
- Based on the [official documentation](https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf "official documentation") for the dataset, we identify the key features to extract for our usage. These are: wind speed, cloud ceiling height, visibility, temperature, dew point, air pressure, precipitation, and snow depth. Snow depth is only available for 200 out of 60,000 individual flights, so we are unable to use it – it is unclear whether some airports just don't report it, or whether it's always zero if unreported.
- For each of these features, multiple bits of info are presented within each column/cell, separated by commas. Generally, these are focused on quality control, and relate to the quality of data presented. We split the data accordingly, remove the data quality measures and keep only the raw data. 
- Further processing is still sometimes required; for example, for the precipitation, different airports report the cumulative precipitation over different time periods (i.e. 1, 3, or even 6 hours), so we need to divide the cumulative precipitation by the reporting time period to somewhat standardize the data.

### 3.	Merging flight and weather data ([merge.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/merge.ipynb "merge.ipynb"))
The weather and flight data is not aligned perfectly, of course. We observe that based on the documentation, the gap between individual weather observations is one hour (although recordings may not necessarily be at the top of the hour). We therefore round both the weather and flight timings (for both departure and arrival) down to the hour and merge the datasets using the `pd.merge` function. We then use interpolation to fill in the missing data points – even within observations (rows of the weather data), certain information may be missing from the raw dataset.

### 4.	Exploratory data analysis ([eda.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/eda.ipynb "eda.ipynb"))
> Note: this file includes detailed visualizations for departure delay, even though our eventual focus was on arrival delay. The file is also quite large, with many different interactive plots, so it will take a while to load and may be slow at times.

The exploratory data analysis revealed some key points about the dataset we are looking at:
- The distribution of the response variable is highly skewed. Most flights are either not delayed or only slightly delayed, and there is a long positive tail with a few flights which are extremely delayed.
- Although there is a weak correlation between several features (i.e. higher previous average delay is related to higher departure delay) and departure delay, it's not practical to design a prediction model for departure delay– at least using the data we have. There is simply too much randomness involved, from passengers arriving late to airports forcing flights to wait for various reasons.
- As expected, there is a strong relationship between departure delay and arrival delay.
- Different months have quite a large difference in the proportion of flights delayed, ranging from 13% in November to 25% in June and July.
- Different times of the day also have a large difference in the proportion of flights delayed, from just 6% for flights departing between 5 and 7 AM, but nearly 30% for flights departing from 8 PM – 9 PM.
- There are other differences between departure/arrival airports and airlines. These are illustrated in the [eda.ipynb notebook](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/eda.ipynb "eda.ipynb notebook").

### 5.	Machine learning: regression ([ml_regr.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/ml_regr.ipynb "ml_regr.ipynb"))
> Note: further preprocessing of flight data (beyond that from `process_flights.ipynb`) is included in this file, as well as in `ml_class.ipynb`, including scaling (with RobustScaler, the best scaling method as many of the input features are highly skewed) and one-hot encoding for categorical variables: origin, destination, and airline.

Here, we apply several regression models to address the subquestion on predicting arrival delay in minutes.
- The baseline case is that `ARR_DELAY = DEP_DELAY`. If a flight is $x$ minutes late to depart, the baseline prediction for arrival is $x$ minutes late as well. 
- We are careful not to use any actual arrival data as input features in the model, which would cause data leakage and defeat the whole purpose. Only information which would be reasonably available at the arrival airport can be used; we think it is okay to use arrival weather data though, as rainfall and weather forecasts are quite accurate in the US (see references) and would be available well in advance of the flight actually arriving.
- We apply several models using a single wrapper function that takes in a model name, and runs it using the given hyperparameters. The same function also re-applies the same model with three different sets of features.
- The first two feature selections rely on the `sklearn.inspection.permutation_importance` function, which randomly shuffles out the input features, and scores feature importance based on how much their removal impacts the quality of the model. We use this function to produce two different feature sets, one with features that have importance at least 3 standard deviations above the mean (which we call `permutation_small`) and one with features which have importance at least 2 standard deviations above the mean (which we call `permutation_big`). 
- The third feature selection uses `sklearn.feature_selection.SelectFromModel` to pick the important features based on an `ExtraTreesClassifier` model.
- Each model is run four times: once with all the features, and once for each of the three feature selection sets. We record the results for $R^2$, median absolute error, and mean absolute error for subsequent comparison. Each of these is also recomputed in an 'adjusted' form, which treats negative predictions and negative actual values as zeroes. In other words, for a flight which is not delayed, a prediction of -20 (20 minutes early) would be considered perfect, just as a prediction of -10 would be. These same metrics are also computed for the baseline, which is discussed above.
- We graph the residual plot as well as the predicted vs. actual plot each time. The [ml_regr.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/ml_regr.ipynb "ml_regr.ipynb") notebook only contains some of these plots, the `regr-plots` folder above contains the rest.
- We apply the following models with the above process: Linear Regression (`LinearRegression`), Huber Regressor (`HuberRegressor`), Linear Support Vector Regression (`LinearSVR`), K-Neighbors Regressor (`KNNRegressor`), Decision Tree Regressor (`DecisionTreeRegressor`), Random Forest Regressor (`RandomForestRegressor`), Gradient Boosting Regressor (`GradientBoostingRegressor`), Multi-Layer Perceptron Regressor (`MLPRegressor`), Stochastic Gradient Descent Regressor (`SGDRegressor`).

### 6.	Machine learning: classification ([ml_class.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/ml_class.ipynb "ml_class.ipynb"))

Here, we apply several classification models to address the subquestion on predicting whether a flight will be late to arrive.
- The baseline case is that `ARR_DELAYED = DEP_DELAYED`. If a flight is late to depart, the baseline prediction for is that it will be late to arrive at its destination as well.
- We attempt random undersampling as well as SMOTE ENN over/undersampling to address the significant class imbalance between the delayed and not delayed flights.
- SMOTE ENN over/undersampling generally resulted in significantly lower model accuracy when compared to random undersampling. Also, since our dataset is relatively large (over 50,000 rows), it may be unnecessary to use the SMOTE ENN method, which might be more preferable for smaller datasets as it preserves a larger number of rows.
- Applying the same feature selection sets from the most accurate regression models (`LinearRegression` with `permutation_small` and `LinearSVR` with `permutation_big`), we used several classification models to predict whether flights will be delayed.
- The evaluation metrics we record for each model run are Recall and F1-score for the non-delayed, as well as the delayed class, classification accuracy, and the area under the Receiver Operating Characteristic (ROC) curve. The same metrics (except the area under the ROC curve) are computed for the baseline case as well. 
- We also plot the labeled confusion matrix and the ROC curve for each model. The [ml_class.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/ml_class.ipynb "ml_class.ipynb") notebook only contains some of these plots, the `class-plots` folder above contains the rest. 
- We apply the following models with the above process: Logistic Regression Classifier (`LogisticRegression`), Decision Tree Classifier (`DecisionTreeClassifier`), Random Forest Classifier (`RandomForestClassifier`), Gradient Boosting Classifier (`GradientBoostingClassifier`), and Multi-Layer Perceptron Classifier (`MLPClassifier`).

### 7.	Machine learning: model evaluation ([eval.ipynb](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/eval.ipynb "eval.ipynb"))
As mentioned earlier, we need to compare the models to the baseline prediction, for both classification and regression.
- For the regression case, two models, the `LinearRegression` and `LinearSVR` performed more or less similarly. This is unsurprising because their implementation is rather similar, but `LinearSVR` additionally imposes a cost on the size of the coefficients. We found `C = 0.5` and `loss = squared_epsilon_insensitive` to be the optimal values of the hyperparameters in the `LinearSVR` model. 
- Nearly all models outperformed the baseline; the best were `LinearRegression` and `LinearSVR`. Other models which also performed quite well include the `GradientBoostingRegressor` and `SGDRegressor`. The table below shows results for a selection of the most accurate regression models.

|                        Model                                                         | Mean abs. error (mins) | Adjusted mean abs. error (mins) | Median abs. error (mins) | $R^2$ score | Adjusted $R^2$ score |
| :------------------------------------------------------------------------------- | :------------------------: | :-------------------------------: | :------------------------: | :-----------: | :------------: |
| **Baseline**                                                                        | 12.140                 | 4.948                           | 10.000                   | 0.878       | 0.933                |
| `LinearRegression` (all features)                                                 | 7.584                  | 3.123                           | 5.643                    | 0.943       | 0.968                |
| `LinearSVR` (`C = 0.5`, `loss = squared_epsilon_insensitive`, all features)   | 7.584                  | 3.123                           | 5.651                    | 0.943       | 0.968                |
| `HuberRegressor` (`epsilon = 2`, `permutation_small` features)                | 7.744                  | 3.198                           | 5.751                    | 0.941       | 0.966                |
| `SGDRegressor` (`penalty = l1`, `alpha: 5e-05`, `permutation_big` features) | 7.837                  | 3.197                           | 5.878                    | 0.940       | 0.966                |
| `GradientBoostingRegressor` (`permutation_big` features)                        | 7.875                  | 3.261                           | 5.949                    | 0.938       | 0.963                |
| `MLPRegressor` (`permutation_big` features)                                     | 8.287                  | 3.286                           | 6.304                    | 0.933       | 0.964                |

- For the classification case, again several models performed similarly well. The best results were achieved by the `LogisticRegression` (standard logistic regression) and `MLPClassifier` (neural network-based) models, listed below. 

| Model  |      F1-score (delayed)      |  F1-score (non-delayed) | Recall (delayed) | Recall (non-delayed) |
|:----------|:-------------:|:------:|:-------------:|:------:|
| **Baseline** |  0.750 | 0.929 |   0.725 | 0.938 |
| `LogisticRegression` (feature set 2) |  0.811 | 0.940 |   0.860 | 0.923 |
| `MLPClassifier` (`alpha = 1e-4`, feature set 1) |  0.816 | 0.943 |   0.852 | 0.930 |

- For a visual representation of these results, please refer to the [eval.ipynb notebook](https://nbviewer.org/github/raghav-n/sc1015-mini-project/blob/main/eval.ipynb "eval.ipynb notebook").
- In general, it is clear that these models are not *significantly* better than the baseline and indeed the baseline is quite accurate as well. This suggests that the departure delay is the most important predictor in judging the arrival delay, which seems unsurprising. The models are able to incorporate other variables to create an even more accurate prediction, though.

## What's new?
What we tried beyond the scope of this course:
- More in-depth data preparation/cleaning: data interpolation, merging datasets, etc.
- Plotly for interactive visualizations
- Incorporating various feature selection methods into each model (based on `permutation_importance` and `SelectFromModel`)
- Random undersampling and SMOTE ENN under/oversampling for imbalanced data
- Several different models for machine learning (for regression: `LinearSVR`, `RandomForestRegressor`, `SGDRegressor`, `HuberRegressor`, and others; for classification: `MLPClassifier`, `GradientBoostingClassifier`)

## Conclusions
Looking at our `LinearRegression` model coefficients, we can make a few conclusions. Flight arrival delay is positively correlated with the following variables:
- Departure delay
- Flight distance
- Taxi-out time
- Average delay for this flight, over the previous 4 days

We also observe that certain airports and airlines are associated with more or less delay based on the coefficients of the model. For example, the `ORIGIN_ATL` and `OP_CARRIER_WN` variables have positive coefficients, suggesting flights departing from Atlanta, as well as Southwest Airlines flights are linked to higher delays, while the `ORIGIN_SFO` and `OP_CARRIER_9E` variables have negative coefficients, suggesting that flights departing from San Francisco and Endeavor Air flights are linked with lower delays.

Thanks for reading!

> **Our roles**<br><br>
	Arunaksh: Data cleaning, merging flight and weather data, presentation<br>
	Jodius: Exploratory data analysis, presentation<br>
	Raghav: Data cleaning, machine learning (regression & classification models)<br>

## References
- _Accuracy, Precision, Recall or F1?_, by Koo Ping Shung (Towards Data Science/Medium): https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
- _Airline Delay and Cancellation Data, 2009 - 2018_, compiled by Yuanyu Mu (Kaggle): https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018
- _API Reference_, imbalanced-learn: https://imbalanced-learn.org/stable/references/index.html
- _API Reference_, scikit-learn: https://scikit-learn.org/stable/modules/classes.html
- _Data Access_, National Centers for Environmental Information: https://www.ncei.noaa.gov/access/search/index
- _Federal Climate Complex, Data Documentation for Integrated Surface Data_, National Centers for Environmental Information: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
- _How Reliable Are Weather Forecasts?_, National Oceanic and Atmospheric Administration: https://scijinks.gov/forecast-reliability/
- _L1 and L2 Regularization — Explained_, by Soner Yıldırım (Towards Data Science/Medium): https://towardsdatascience.com/l1-and-l2-regularization-explained-874c3b03f668
- _StandardScaler, MinMaxScaler and RobustScaler techniques – ML_, by Ashwin Sharma P (Geeks for Geeks): https://www.geeksforgeeks.org/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/
- _User Guide_, scikit-learn (https://scikit-learn.org/stable/user_guide.html)
