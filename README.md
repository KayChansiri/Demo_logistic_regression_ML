# Logistic Regression, Sigmoid Function, the Concept of Likelihood Estimation, and More

In my previous post, I discussed linear regression from a machine learning (ML) perspective. Today, let’s delve into a different type of regression used to predict binary outcomes — logistic regression. In this post, I will also discuss important concepts relevant to the algorithm, such as the sigmoid function, log-likelihood estimation, class imbalance adjustment, and more. If you are ready, let's get started!

## Why Logistic Regression?

If you're like me, the first time you learn about logistic regression, you may wonder, "Why can’t we use linear regression to predict a binary outcome (e.g., outcomes of which levels are 0 or 1, "True" or "False", etc.)?" The answer lies in the limitations of linear regression for binary outcomes. Linear regression predictions can extend beyond the [0, 1] range, leading to unreasonable interpretations when our target outcome is strictly within this boundary. Thus, using an alternative model that works well with binary decisions is better. Logistic regression, which uses the sigmoid function, is the answer.

Before we get to the sigmoid function, there are three key terms you should familiarize yourself with in logistic regression: probability, odds, and logit: 

* **Probability** refers to the chance of an event occurring, ranging from 0 to 1, and can be interpreted as a percentage. For example, if the probability of a customer subscribing to a new streaming service is 0.6, it means there is a 60% chance the customer will subscribe.
* **Odds** are calculated by dividing the probability of an event occurring by the probability of it not occurring.
* **Logit** is the natural logarithm of the odds. The term is also referred to as log odds. See below.


<img width="601" alt="Screen Shot 2024-07-10 at 3 19 09 PM" src="https://github.com/KayChansiri/Demo_logistic_regression_ML/assets/157029107/b69425ca-1c16-40f5-94fd-858ac388df1b">

Mathematically, you can convert logit back to odds by applying the exponential function, and you can convert odds to probability by dividing the odds by 1 plus the odds:

<img width="477" alt="Screen Shot 2024-07-25 at 8 49 48 AM" src="https://github.com/user-attachments/assets/543c3f45-9923-43dc-8f20-c0a06b9d3e6f">


I know these concepts might sound confusing at first, but bear with me. I promise that by the end of this post, you'll have a  better understanding. Let’s get back to the question I asked previously: why do we use the sigmoid function and why logistic regression for  binary outcomes? 

## Sigmoid Function

The formula below represents the sigmoid function:

<img width="151" alt="Screen Shot 2024-07-25 at 9 00 00 AM" src="https://github.com/user-attachments/assets/c3e774ea-846e-4ca3-8e66-20910860893d">

According to the function formula, whatever value is plugged into the function, the output is **probabilities** and always bounded between 0 and 1.


<img width="413" alt="Screen Shot 2024-07-25 at 9 01 49 AM" src="https://github.com/user-attachments/assets/f45fd0d2-a409-4dbc-96dd-abf71ea6c487">

Imagine *x* is your typical linear regression function that could be represented by B<sub>0</sub> +  B<sub>1</sub>X<sub>1</sub>. When you plug in the value into the formula, the probability of detecting the event of your interest, which is mostly represented by *y* = 1, would be equal to:


<img width="330" alt="Screen Shot 2024-07-25 at 9 18 05 AM" src="https://github.com/user-attachments/assets/0d8fd29e-9fbc-490f-841d-ee55de674296">

Let’s take a look at an example. Say you want to predict the probability of a customer subscribing to a streaming service (1) versus no-subscription (0). Your predictor is customers’ age. Keep in mind that in the real world, we will likely have more than one predictor, but I will use only one predictor here for simplicity in demonstration. Suppose that your intercept is B<sub>0</sub> = -3 and the coefficient of age is B<sub>1</sub> = 0.1. Assume that customer A is 25 years old. You plug in all of the values into the sigmoid function above, and the probability of the customer subscribing to the streaming service is calculated as the following.

The linear regression function X can be represented as:

X = B<sub>0</sub>+ B<sub>1</sub>⋅Age

x= −3 + 0.1⋅25

x= −0.5

Now, plug this value into the sigmoid function:

<img width="246" alt="Screen Shot 2024-07-25 at 9 27 43 AM" src="https://github.com/user-attachments/assets/2aa6f133-4a05-41f1-9a84-13cc5b129a37">

You will get P(*y*=1) ≈ 0.3775. So, the probability of the customer subscribing to the streaming service is approximately 37.75%.


Now imagine doing this with all customers in the dataset. You will get a plot like the one below:

<img width="459" alt="Screen Shot 2024-07-25 at 9 30 29 AM" src="https://github.com/user-attachments/assets/76275306-bfc2-43fc-8197-e15972104a5e">

One challenge that we face if we end our logistic regression work here with the sigmoid function is the challenge in interpretation. Since the line does not follow a linear relationship due to the application of the sigmoid function, a one-unit increase in *X* does not correspond to a one-unit increase in *Y*. To make our interpretation easier, we convert the sigmoid function into logit, which is mathematically equivalent and simplifies the interpretation of the results. Still confused? 

To better explain, the outcome of a signmoid function is in the form of probability, ranging between 0 and 1, according to this formula: 

<img width="183" alt="Screen Shot 2024-07-25 at 9 41 16 AM" src="https://github.com/user-attachments/assets/fed0d7ad-e4b5-4467-b3e1-c3f5f322f4ed">

The logit function does the opposite: it takes a probability (a number between 0 and 1) and converts it back into a linear combination of predictors. This is done by taking the natural logarithm of the ratio of the probability of success to the probability of failure (i.e., the odds):

<img width="198" alt="Screen Shot 2024-07-25 at 9 42 39 AM" src="https://github.com/user-attachments/assets/8687d9cb-20c1-4aa7-941d-c81944da985c">

Remember I said earlie that *X* can be represented by your typical linear regresion function. Thus, converting a sigmoid function to logit for an easier interpretation of the output results in the following equation: 

<img width="281" alt="Screen Shot 2024-07-25 at 9 45 04 AM" src="https://github.com/user-attachments/assets/1477c19e-e560-4b1a-9a96-e95f40129563">

By converting the probability to the logit (i.e.,log-odds), we transform the nonlinear relationship into a linear one, making it easier to interpret. The coefficients in the logit model tell us how a one-unit change in a predictor affects the log-odds (i.e., logit) of the outcome.

Even though we solve the linear interpretation problem, trying to understand what ‘one unit increase in logit’ means exactly is still challenging. Thus, we often convert regression coefficients to something easier for interpretation, like odds ratios. This can be done easily by exponentiating the coefficient.

For instance, according to the age and customers' subscription example I mentioned previously (the intercept = -3 and the slope = 0.1), we can say that when age is equal to zero, the odds of subscribing to the service is e<sup>-3</sup> ≈ 0.05. For every additional year of age, the odds of subscription increase by approximately e<sup>0.1</sup> ≈ 1.105, which means the odds are multiplied by 1.105, or increased by about 10.5%. 
Putting it together, when combining both the intercept and the coefficient for customer A who is 25 years old, x=−3+0.1⋅25=−0.5. The probability of subscription is: 

<img width="227" alt="Screen Shot 2024-07-25 at 10 22 19 AM" src="https://github.com/user-attachments/assets/77120fc8-57ad-4b85-81ff-c069f49af055">

The odds of subscribing to the service when the customer is 25 years old are e<sup>-0.5</sup> ≈ 0.607..

**Note that when the coefficient is positive, we use e<sup>coefficient</sup> to get the odds ratio, and the output will be more than 1. When the coefficient is negative, e<sup>coefficient</sup>still gives us the odds ratio, but it indicates a decrease in odds as the output will be less than 1.**

In the real world, we tend to have more than one predictor. We can write the logit formula of logistic regression as:

<img width="524" alt="Screen Shot 2024-07-25 at 10 24 18 AM" src="https://github.com/user-attachments/assets/dfbad6f9-9a24-48c7-93a3-ef20422ae7e6">


I hope now you can see that how are probabilities (using the sigmoid function such that the output values are bounded between 0 and 1), logit (i.e., log odds), and odds ratios used in logistic regression functions. In conclusion, we first started with applying a sigmoid function to a typical linear regression so that our output values represent reality by being bounded between 0 and 1. As it's challenging to interpret how a one-unit increase in *X* would result in how many units increase in *Y* for a nonlinear function (i.e., the sigmoid function), we convert the function to logit or log odds, which is a linear function. Nonetheless, trying to understand how a one-unit increase in *X* would result in how many logit increases in *Y* is still challenging for us humans, so we convert the logit to odds ratios.
The process can be mathematically reverted as well to get the probabilities from odds ratios.

## The Concept of Likelihood 

In my previous post about linear regression, I showed you a visualization of how a software program that you use to run regression comes up with a set of beta coefficients (e.g., by relying on matrix operations or gradient descent boosting). For logistic regression, things work a bit differently. To get the best set of regression coefficients, we use the concept of likelihood. Let’s try to understand the basic idea of this concept first.

Say you work for a streaming service company based in Northern Virginia, where the Asian population is on the rise, and you assume that the probability of Asian customers subscribing to a new streaming service from Korea should be quite high, around 0.8. In other words, we can say *p* = 0.8, meaning that there is an 80% chance that a customer would subscribe to the service. Then you look at the actual data and observe that at least 7 out of 10 customers subscribe to the streaming service. This observed data could be represented by the vector below:

Y=(H,H,H,T,H,H,H,T,H,T)

According to the vector, H=subscription, and T =no subscription. Now, when plugging the 0.8 probability into the vector, you would get the following likelihood:

*L*(*Y*∣*p*=0.8) = 0.8×0.8×0.8×(1−0.8)×0.8×0.8×0.8×(1−0.8)×0.8×(1−0.8) ≈ 0.001677

Thus, the likelihood of observing the data given that the probability of subscription is 0.8 is approximately 0.001677. In other words, we can say that if we assume that the probability is 0.8, the likelihood of observing the outcome in our dataset (7 subscriptions and 3 no subscriptions) is 0.001677.

If you think that the probability of an Asian customer subscribing to the streaming service might be a bit lower, like about 0.5 due to the economic recession, the likelihood would be:

*L*(*Y*∣*p*=0.5)=0.5×0.5×0.5×(1−0.5)×0.5×0.5×0.5×(1−0.5)×0.5×(1−0.5) = 0.0009765625

According to the two likelihood estimations above, we can say that an estimate of *p* being equal to 0.8 (likelihood = 0.001677) is more likely than an estimate of *p* being equal to 0.5 (likelihood = 0.0009765625). In other words, our observation of 7 subscriptions and 3 no-subscriptions is more likely if we estimate *p* as 0.8 rather than 0.5.

## Maximum Likelihood Estimation (MLE)

Now you may have a question regarding which *p* you should use such that you get the highest likelihood that best reflects the actual observed data (7 subscriptions and 3 no-subscriptions). The answer is you can try different values of *p* from 0 to 1 and see which one yields the highest likelihood as seen in the plot below:

<img width="925" alt="Screen Shot 2024-07-25 at 12 46 12 PM" src="https://github.com/user-attachments/assets/6d767a44-489c-4eb5-a3eb-7aa8911e411d">

In the plot, the x-axis represents the probability of observing a subscription (H), and the y-axis indicates the likelihood of observing 7 subscriptions (H) and 3 no-subscriptions (T). The peak probability value here, about 0.7, yields the highest likelihood (i.e., about 0.0022) of observing 7 subscriptions and 3 no-subscriptions. Therefore, the maximum likelihood estimate of the probability of observing a subscription for this particular dataset is 0.7 given the 10 observations we have made.

Note that the concept of likelihood I mentioned above should work fine if you have only 10 observations. However, in the real world, you tend to have many more observations, like thousands to millions. Keeping multiplying *p* for each participant together could lead to high computational complexity. This is where the concept of log likelihood could be helpful.

At the end, the probability that maximizes likelihood is also the same number that maximizes the log likelihood. Thus, it does not matter that much if we use log likelihood instead of likelihood. The formula of log likelihood based on the concept of likelihood is below:

<img width="472" alt="Screen Shot 2024-07-25 at 12 51 55 PM" src="https://github.com/user-attachments/assets/9997a45a-804e-41dc-b5c9-32ca689abbfb">

* *Y*<sub>*i*</sub> is the observed outcome (1 for subscription, 0 for no subscription).
* *p* is the probability of subscribing.
* *L*(*Y*∣*p*) is the likelihood of observing the data given the probability *p*.


## Maximum Likelihood Estimation (MLE) for Logistic Regression

Now that you have learned about the concept of MLE, let’s see how we can apply the concept with logistic regression. Let’s take another look at the logistic regression equation that I introduced to you previously:

<img width="321" alt="Screen Shot 2024-07-25 at 1 56 57 PM" src="https://github.com/user-attachments/assets/ede755ea-a32a-4449-aece-242a5bbba38c">

Imagine that at first, we have a random set of coefficients B<sub>0</sub>=−0.3 and  B<sub>1</sub> = 0.1 as discussed previously. When we plug in the values and each *X* into the equation above, we get the logit for every observation. Note that unlike the example above, where I simply said that the probability of an Asian customer subscribing to the streaming service is 0.8, in reality, each sample should have a different *X* (such as age) and therefore should have a different probability.

For instance, say each customer has a different probability as shown in the data table below:

<img width="530" alt="Screen Shot 2024-07-25 at 2 03 28 PM" src="https://github.com/user-attachments/assets/559dcb5f-6990-48d7-8d00-a33969d99a7c">

In this table:
* Age is the predictor variable *X*.
* Subscription (Y) indicates whether the customer subscribed (1) or not (0).
* Probability (P) is calculated using the logistic regression model with the initial coefficients B<sub>0</sub>0=−0.3 and  B<sub>1</sub>0 = 0.1

You can calculate these probabilities using the logistic regression equation:

<img width="212" alt="Screen Shot 2024-07-25 at 4 12 28 PM" src="https://github.com/user-attachments/assets/0e47ec08-6be1-4c4d-a31c-d5a7d5d63f0d">

Using the above probabilities, we can calculate the log likelihood for the entire dataset using the formula below I mentioned previously: 

<img width="460" alt="Screen Shot 2024-07-25 at 4 14 21 PM" src="https://github.com/user-attachments/assets/aca728b0-6f54-4e3e-865e-54166d2a580c">

Plugging in the values in the dataset above, we would get the log likelihood about -28.29.

Just like in linear regression, one question to ask is which sets of beta coefficients would provide a higher likelihood of the data. The answer is pretty much the same. We try different sets of Betas and see which ones yield the highest log likelihood. However, unlike linear regressions where you are trying to find the best beta coefficients that yield the lowest SSE (Sum of Squared Errors), here you find the best coefficients that yield the highest log likelihood.

The way we do this is called the logistic loss function, which basically refers to trying to find the set of coefficients in a model that can obtain the maximum likelihood estimates of the coefficients for the logistic regression model.


<img width="711" alt="Screen Shot 2024-07-25 at 4 36 11 PM" src="https://github.com/user-attachments/assets/53162270-f189-4d35-95a6-50dcff2dcced">

## Regularization and Evaluation Metrics in Logistic Regression

The way that regularization works for logistic regression is quite similar to how it works in linear regression, which involves adding penalty terms to the loss function to avoid large coefficients. We have different types of regularization, including ridge, lasso, and elastic net. You may refer to my [previous post](https://github.com/KayChansiri/Demo_Linear_Regression_ML) to read more about regularization techniques in regression functions.

Although logistic and linear regression share a similar regularization process, the evaluation metrics are different and emphasize accuracy, precision, recall, F1 scores, and the area under the curve (AUC). Read more in [the post](https://github.com/KayChansiri/Demo_Performance_Metrics) I wrote previously regarding evaluation metrics for binary outcomes. 

## Class Imbalance

In the real world, we do not always have projects where the outcome is balanced between the two categories. For example, consider fraud detection in banking; the percentage of fraudulent transactions is likely much lower than non-fraudulent transactions. The imbalance of the target outcome classes can influence the model performance. I wrote a post about which metrics are better when we have class imbalance [here](https://github.com/KayChansiri/Demo_Performance_Metrics). In addition to selecting the right metric to evaluate model performance, there are some strategies that could help to boost the performance of logistic regression when dealing with class imbalance.

### 1. SMOTE (Synthetic Minority Over-sampling Technique)
* **What it is**: SMOTE is a technique used to generate synthetic samples for the minority class to balance the class distribution.
* **When to use it**: Use SMOTE when you have a significant class imbalance and need to improve the performance of your model by creating a more balanced dataset.
* **When not to use it**: Avoid using SMOTE if the minority class has very few instances, as the synthetic samples might not represent the actual data distribution well. From my personal experience, if the ratio of your majority: minority class is less than 100:1, most of the times, SMOTE will not work well.

### 2. Undersampling the Majority Class
* **What it is**: Undersampling involves reducing the number of instances in the majority class to balance the dataset.
* **When to use it**: Use undersampling when you have a large dataset and can afford to lose some majority class instances without significantly impacting the model's ability to learn. However, make sure that your majority instances follow a normal distribution. Otherwise, you may have a biased sample when you perform the undersampling technique, although you can write a function to randomly select majority class instances to represent the distribution of their population pool.
* **When not to use it**: Be cautious when the class distribution ratio is crucial to the problem at hand. For example, in fraud detection, if fraudulent transactions (coded as 1) make up only 5% of the dataset, undersampling the majority class (non-fraudulent transactions coded as 0) to make a 1:1 ratio does not reflect the real-world distribution and could influence the validity of your project findings.

### 3. Weight Adjustment
* **What it is**: Adjusting the weights involves assigning more weight to the minority class during model training to penalize misclassifications of the minority class more heavily.
* **When to use it**: Use weight adjustment when you want to give more importance to the minority class without altering the dataset.
* **When not to use it**: Avoid using weight adjustment if the model already handles class imbalance well, as it might lead to overfitting.

### 4. Other Techniques
* There are other techniques to address class imbalance, such as using different algorithms that are robust to class imbalance, employing ensemble methods (e.g., random forest), or leveraging anomaly detection techniques. Each method has its advantages and should be considered based on the specific context of the problem.

> Another thing you may wonder if you have class imbalance is whether you should apply regularization first or the class adjustment techniques mentioned above first. The answer is to apply the class adjustment techniques first. This helps to address the imbalance in the data, allowing the regularization process to work more effectively by focusing on a balanced dataset.

## Example

Let's take a look at a real-world example. The dataset I use today discusses subscription to a streaming service among Asian Americans. This fictional streaming company cares about detecting all potential customers who may subscribe to their service and wants to reach out to as many of them as possible to boost sales rates. The variables are:

* **user_id**: A unique identifier for each user.
* **days_since_signup**: The number of days since the user signed up for the streaming service.
* **age**: The age of the user.
* **payment_activity**: Records of the user's payment activities.
* **login_activity**: Records of the user's login activities.
* **content_view_activity**: Records of the user's content view activities.
* **app_update_activity**: Records of the user's app update activities.
* **customer_service_activity**: Records of the user's interactions with customer service.
* **account_change_activity**: Records of the user's account changes.
* **error_report_activity**: Records of error reports submitted by the user.
* **content_download_activity**: Records of the user's content download activities.
* **other_activity**: Records of other miscellaneous activities.
* **subscription_start_date**: The date when the user's subscription started.
* **account_issue_history**: Records of any issues with the user's account.
* **unsubscribe_reason_email**: Indicates if the user unsubscribed due to email reasons.
* **unsubscribe_reason_ad**: Indicates if the user unsubscribed due to ads.
* **unsubscribe_reason_content**: Indicates if the user unsubscribed due to content-related reasons.
* **unsubscribe_reason_ui**: Indicates if the user unsubscribed due to user interface issues.
* **unsubscribe_reason_login**: Indicates if the user unsubscribed due to login issues.
* **unsubscribe_reason_registration**: Indicates if the user unsubscribed due to registration issues.
* **gender_male**: Indicates if the user is male.
* **activity_count**: The total count of user activities.
* **session_count**: The total number of user sessions.
* **subscription_status**: The binary outcome indicating whether the user is currently subscribed (1) or not (0).
* **race_asian**: Indicates if the user is Asian.

### Data Preparation 

Before we start with any machine learning techniques, let's clean and check assumptions to ensure that the data is suitable for logistic regression modeling. Since this is a synthetic dataset, there are no missing values. However, in the real world, you would need to handle missing values first1..

**1. Standardizing Continuous Variables**

This step is performed such that it is easier to interpret and compare the effect sizes of each continuous predictor ('age', 'days_since_signup', 'activity_count', 'session_count') on the outcome: 

```ruby
#standardize continuous variables 
from sklearn.preprocessing import StandardScaler

# Select the columns to standardize
columns_to_standardize = ['age', 'days_since_signup', 'activity_count', 'session_count']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
data[columns_to_standardize] = scaler.fit_transform(data[columns_to_standardize])

```

**2. VIF**

The next step is to check for multicollinearity using the Variance Inflation Factor (VIF).

```ruby
#Check VIF
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Function to calculate VIF
def calculate_vif(data):
    vif = pd.DataFrame()
    vif["Variable"] = data.columns
    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif


# Remove non-predictor columns
X = data.drop(columns=['subscription_start_date', 'subscription_status', 'user_id'])

# Check VIF
vif = calculate_vif(X)
print(vif)

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Define the target outcome
y = data['subscription_status']


```

Here is the output: 


<img width="410" alt="Screen Shot 2024-07-29 at 1 48 20 PM" src="https://github.com/user-attachments/assets/6e1fbdb9-800c-4bc7-a9de-c8822e5ad31b">

According to the output, 'content_view_activity' and 'unsubscribe_reason_ad' have high VIF values, indicating that these variables may be highly correlated with one another. Therefore, I will drop 'content_view_activity' from the analysis from now on.

**3. Logistic regression assumption check** 

**3.1 Linearity between Continuous Predictors and Log-Odds of the Outcome**: Before fitting the logistic regression model, it is important to check the assumption that there is a linear relationship between the continuous predictors and the log-odds of the outcome:

```ruby
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Add a small constant to avoid division by zero
data['logit'] = np.log(data['subscription_status'] / (1 - data['subscription_status'] + 1e-10))

# Plot logit vs continuous predictors
for predictor in ['age', 'days_since_signup', 'activity_count', 'session_count']:
    plt.scatter(data[predictor], data['logit'])
    plt.xlabel(predictor)
    plt.ylabel('Logit (log odds)')
    plt.title(f'Logit vs {predictor}')
    plt.show()
```

Here is the output: 

<img width="634" alt="Screen Shot 2024-07-29 at 2 06 00 PM" src="https://github.com/user-attachments/assets/636f3f9b-f925-481f-bb79-951e70caa901">


<img width="601" alt="Screen Shot 2024-07-29 at 2 06 32 PM" src="https://github.com/user-attachments/assets/abad0b76-a638-4889-99dd-8469370c36e7">


<img width="597" alt="Screen Shot 2024-07-29 at 2 06 39 PM" src="https://github.com/user-attachments/assets/a19190df-f2ea-4652-bfbc-6434d07ac7eb">

<img width="599" alt="Screen Shot 2024-07-29 at 2 06 46 PM" src="https://github.com/user-attachments/assets/05cca5d4-0de8-4731-a7ec-4284baaff7fb">

The output plots show constant y-values (logit) across different x-values (predictors), which suggests one of the following: 1) The continuous variables do not influence the outcome, or 2) The continuous predictors are not linearly related to the outcome. I have run logistic regression using these predictors separately and found that they significantly predict the outcome. Thus, the second scenario is more likely: the predictors are not linearly related to the log-odds of the outcome.

To address the non-linearity between the predictors and the log-odds of the outcome, consider the following solutions: Transform the continuous predictors by applying logarithm, square root, or polynomial terms to the predictors to capture the non-linear relationships, and explore the interactions of the continuous predictors with other features, which should be informed by domain theories and knowledge.

Since this is synthetic data and I have not conducted any literature review regarding significant interaction predictors of customer subscription, I will omit the interaction method. I will also omit the log transformation method, as certain standardized variables have negative values, resulting in infinite values after log transformation. Lastly, I will omit the square root method, as it does not work with negative numbers.

For this project, I will perform only the polynomial transformation method to see if adding polynomial terms of these continuous predictors may improve the model performance, as these predictors are suspected to not have a linear relationship with the outcome.

```ruby
# Polynomial terms
data['age_squared'] = data['age'] ** 2
data['days_since_signup_squared'] = data['days_since_signup'] ** 2
data['activity_count_squared'] = data['activity_count'] ** 2
data['session_count_squared'] = data['session_count'] ** 2
```

Now we will run a simple logistic regression model from a traditional statistical perspective without performing any machine learning techniques. This will help us see if adding the polynomial terms actually improves the model fit. This step ensures that the predictors we include in the model during the machine learning phase are meaningful and do not lead to overfitting.

```ruby
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2


# Features for the model without polynomial terms
features_model = ['days_since_signup', 'age', 'payment_activity', 'login_activity', 
                  'app_update_activity', 'customer_service_activity', 'account_change_activity', 'error_report_activity',
                  'content_download_activity', 'other_activity', 'account_issue_history', 'unsubscribe_reason_email',
                  'unsubscribe_reason_ad', 'unsubscribe_reason_content', 'unsubscribe_reason_ui', 'unsubscribe_reason_login',
                  'unsubscribe_reason_registration', 'gender_male', 'activity_count', 'session_count', 'race_asian']

# Features for the model with polynomial terms
features_model_poly = features_model + ['age_squared', 'days_since_signup_squared', 'activity_count_squared', 'session_count_squared']

# Fit logistic regression model without polynomial terms
X = data[features_model]
X = sm.add_constant(X)
logit_model = sm.Logit(data['subscription_status'], X).fit()

# Fit logistic regression model with polynomial terms
X_poly = data[features_model_poly]
X_poly = sm.add_constant(X_poly)
logit_model_poly = sm.Logit(data['subscription_status'], X_poly).fit()

# Model summaries
print(logit_model.summary())
print(logit_model_poly.summary())

# Model Comparison
# AIC and BIC comparison
print(f"Model without polynomial terms AIC: {logit_model.aic}, BIC: {logit_model.bic}")
print(f"Model with polynomial terms AIC: {logit_model_poly.aic}, BIC: {logit_model_poly.bic}")

# Likelihood Ratio Test
llf_full = logit_model_poly.llf
llf_reduced = logit_model.llf
lr_stat = 2 * (llf_full - llf_reduced)
df_diff = logit_model_poly.df_model - logit_model.df_model
p_value_lr = chi2.sf(lr_stat, df_diff)
print(f"Likelihood Ratio Test Statistic: {lr_stat}, p-value: {p_value_lr}")
```

Here is the output: 

<img width="672" alt="Screen Shot 2024-07-30 at 10 18 52 AM" src="https://github.com/user-attachments/assets/6fffdb4e-6923-47a2-b7a2-2172735907f8">



The model with polynomial terms has lower AIC and BIC values compared to the model without polynomial terms, suggesting that it provides a better fit to the data despite its additional complexity. Additionally, the p-value is extremely small (3.29×10⁻⁹), indicating strong evidence that the model with polynomial terms fits the data significantly better than the model without polynomial terms.

**3.2: Outliers:** I used Cook's distance to calculate the outliers. The output below indicates that none of the Cook's distance values are greater than 1, thus there are no influential outliers, despite some instances having more extreme Cook's scores than others. However, considering the dataset is quite large, I did not deal with these potential outliers.

```ruby
influence = logit_model_poly.get_influence()
cooks_d = influence.cooks_distance[0]

# Plot Cook's distance
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.xlabel('Index')
plt.ylabel("Cook's Distance")
plt.show()

```

The output: 

<img width="600" alt="Screen Shot 2024-07-29 at 2 40 25 PM" src="https://github.com/user-attachments/assets/f8bd062a-b679-41d6-9236-58818f4879cc">

**3.3 Homoscedasticity**: Homoscedasticity means that the variance of the residuals is constant across all levels of the independent variables. In the context of logistic regression, we assess this by examining the spread of the predicted probabilities. A good output plot should show a consistent spread of resiauls (i.e., the distance between observed versus predicted probabilities) across all levels of the predicted values with an even spread, random distribution, and no funnel looking shapes. 


```ruby
# Fit logistic regression model with polynomial terms
X_poly = data[features_model_poly]
X_poly = sm.add_constant(X_poly)
logit_model_poly = sm.Logit(data['subscription_status'], X_poly).fit()

# Get residuals from the model
residuals = logit_model_poly.resid_response

# Get fitted values (predicted probabilities)
fitted_values = logit_model_poly.predict(X_poly)

# Plot residuals vs fitted (i.e., predicted) values
plt.scatter(fitted_values, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

```
The output: 


<img width="631" alt="Screen Shot 2024-07-29 at 2 51 06 PM" src="https://github.com/user-attachments/assets/bfe4fa4f-6381-4fa6-8403-af10d0d96f24">


According to the output, the distribution of the predicted probabilities is not random and forms a specific shape, indicating that the homoscedasticity assumption is not met. What we should do is apply transformations or consider different modeling techniques to address this issue. However, for simplicity in this demonstration, I will skip these steps and continue with the machine learning process.

### Machine Learning 

Now that we have prepared the data, let's begin by finding the best method to deal with class imbalance first. In this dataset, we have a pretty severe class imbalance as only 739 customers actually subscribe to the service, while 22,834 participants do not. We should always deal with class imbalance first before performing any regularization techniques to ensure that the model accurately learns from both classes and doesn't become biased towards the majority class (i.e., people who do not subscribe to the service in the current example). Let's begin with the baseline model without any class imbalance adjustment first.


**1. Baseline model**
```ruby
#1. Logistic regression baseline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Assuming X and y are your feature matrix and target variable respectively

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Fit logistic regression without any class imbalance adjustment
log_reg_baseline = LogisticRegression(max_iter=10000, random_state=42)
log_reg_baseline.fit(X_train, y_train)

# Predict and evaluate
y_pred_baseline = log_reg_baseline.predict(X_test)
print("Baseline Logistic Regression (without SMOTE)")
print(classification_report(y_test, y_pred_baseline))
print("AUC:", roc_auc_score(y_test, y_pred_baseline))
```
The output: 

<img width="472" alt="Screen Shot 2024-07-30 at 10 34 21 AM" src="https://github.com/user-attachments/assets/c032515b-0f93-46ea-9a15-aab830ef1c63">

The recall for the minority class (subscribers) is only 1%, which is not surprising given the severe class imbalance, with the ratio of majority to minority class approximately 30:1. This recall indicates that the model correctly identifies only 1% of the actual subscribers. Among those predicted as subscribers, 20% are correct predictions (i.e., precision).

Note that while the overall accuracy is quite high at 97%, this does not guarantee that the model performs well in predicting the minority class. The high accuracy is driven by the high recall and precision for predicting non-subscribers (i.e., the majority class), which is not our primary group of interest. The area under the curve (AUC) is 50%, indicating that the model has a limited ability to distinguish between subscribers and non-subscribers.

**2. SMOTE**

```ruby
#2. Simple logistic regression with SMOTE 

from imblearn.over_sampling import SMOTE

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Fit logistic regression
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train_smote, y_train_smote)

# Predict and evaluate
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred))

```
The output: 

<img width="469" alt="Screen Shot 2024-07-30 at 10 21 59 AM" src="https://github.com/user-attachments/assets/f4611243-b14b-41f1-941e-75c03a24da2e">

According to the output from the SMOTE class imbalance adjustment, the recall for the minority class (subscribers) is 30%, indicating that the model correctly identifies 30% of the actual subscribers. This percentage is much better than the baseline model, where I did not create synthetic minority classes. Among those predicted as subscribers, only 6% are correct predictions (i.e., precision). Note that the precision has significantly dropped from the baseline model (20%). This is because the model is now more likely to predict subscribers, increasing the number of false positives.

Like the baseline model, the overall accuracy of the SMOTE model is quite high at 85%. However, this does not guarantee that the model performs well in predicting the minority class. The high accuracy is driven by the high recall and precision for predicting non-subscribers (i.e., the majority class). The area under the curve (AUC) is 58%, indicating that the model still has a limited ability to distinguish between subscribers and non-subscribers.
      
**3. Logistic regression with class weight adjustment**

In the code below, the class_weight='balanced' parameter automatically adjusts the weights inversely proportional to class frequencies in the input data to help the model pay more attention to the minority class during training. You may have a question: Are the weights applied to the majority or minority class? The answer is they are applied to both classes but with different magnitudes. For the minority class, the weight is higher to compensate for its underrepresentation. For the majority class, the weight is lower to avoid the model being biased towards it. The goal is to balance the influence of both classes on the model training process. You can manually adjust the weight for each class or you can just set the class_weight = 'balanced' as I did below to let the algorithm automatically calculate the class weights inversely proportional to the class frequencies using the formula: 

<img width="140" alt="Screen Shot 2024-07-30 at 10 54 04 AM" src="https://github.com/user-attachments/assets/160196cf-8836-44f9-b025-9267470ebbd6">

* *n* is the total number of samples.
* *k* is the number of classes.
* *n<sub>i</sub>* is the number of samples *n* class *i*.

```ruby
#3 logistic regression with class weight adjustment
log_reg_weighted = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)
log_reg_weighted.fit(X_train, y_train)

# Predict and evaluate
y_pred_weighted = log_reg_weighted.predict(X_test)
print("Weighted Logistic Regression")
print(classification_report(y_test, y_pred_weighted))
print("AUC:", roc_auc_score(y_test, y_pred_weighted))
```
The output: 
<img width="473" alt="Screen Shot 2024-07-30 at 10 56 28 AM" src="https://github.com/user-attachments/assets/b1fec721-0307-4be0-a4a6-e0d191be7ddf">

 Recall for the minority class (subscribers) is 74%, which means the model correctly identifies 74% of the actual subscribers. This is a significant improvement over both the SMOTE technique and the baseline model. However,precision for the minority class is only 8%, indicating that among the predicted subscribers, merely 8% are actually correct. This low precision is because the model, while being effective at identifying subscribers, also incorrectly classifies many non-subscribers as subscribers, resulting in a high number of false positives. Overall accuracy is 77%, which is lower than the baseline model but more balanced in terms of performance across both classes. AUC is 75%, indicating that the model has a reasonable ability to distinguish between subscribers and non-subscribers, suggesting a better overall performance than the baseline and SMOTE models. 

While the precision is low, the recall is quite high, which means the model is effective at identifying potential subscribers. As mentioned earlier, this fictional streaming company cares more about detecting all potential customers who may subscribe to the service and wants to reach out to as many of them as possible to boost sales rates, the current recall rate may be satisfactory enough. We can boost the precision by adjusting the decision threshold (using a value other than the default 0.5) as discussed in [my previous post](https://github.com/KayChansiri/Demo_Performance_Metrics). However, given the company's goal to maximize recall, the current approach is appropriate.


**4. Logistic regression with undersampling**
For this method, we will try undersampling the majority class. In the code below, the ratio of undersampling is determined by how you configure the RandomUnderSampler. By default, the RandomUnderSampler will balance the minority and majority classes to have the same number of instances (1:1). However, you can specify the desired ratio by using the sampling_strategy parameter.

The ratio that you specify should also reflect the true population's distribution. From my quick search on Perplexity, as of January 2024, streaming made up 45.4% of Asian Americans' TV usage, compared to 36.0% for the general U.S. population. Based on this information, we will set the ratio for undersampling to reflect a similar distribution. If we assume the minority class represents the subscribers among Asian Americans and the majority class represents the non-subscribers, we might set a ratio close to this distribution. For simplicity, let's assume a custom ratio of approximately 45.4% subscribers to 54.6% non-subscribers, which translates to a 0.83 ratio (45.4/54.6).

```ruby
#4 undersampling
from imblearn.under_sampling import RandomUnderSampler


# Apply undersampling to the training data with a custom ratio (0.83:1)
undersampler = RandomUnderSampler(sampling_strategy=0.83, random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Fit logistic regression
log_reg_under = LogisticRegression(max_iter=1000, random_state=42)
log_reg_under.fit(X_train_under, y_train_under)

# Predict and evaluate
y_pred_under = log_reg_under.predict(X_test)
print("Undersampling Logistic Regression (0.83:1 ratio)")
print(classification_report(y_test, y_pred_under))
print("AUC:", roc_auc_score(y_test, y_pred_under))
```
The output:

<img width="473" alt="Screen Shot 2024-07-30 at 10 56 28 AM" src="https://github.com/user-attachments/assets/57a6e2b5-3c16-4913-a13e-5d4f409c0971">

The undersampling approach (0.83:1 ratio) achieves a better recall for the minority class (56%) compared to the baseline and SMOTE models. However, the recall rate is lower than that of the class weight adjustment model. The precision (7%) is also low, but since this metric is not of the company's primary interest, we will ignore it for now. The AUC of 67% indicates that the model has a reasonable discriminative ability, but there is room for improvement.

Among all the methods I have tried, the class weight adjustment is the best at achieving the highest recall. Thus, we will proceed with using this method for the next step: regularization.

### Regularization 

**1. Lasso regression** 
If you are not familiar with different types of regularization, refer back to my [previous post](https://github.com/KayChansiri/Demo_Linear_Regression_ML). The first method that I use here is lasso, which forces certain high coefficients to be exactly zero. In the code below, I use lasso, along with class weight adjustment to deal with class imbalance. I also use cross-validation with five folds. In each fold, the model tests multiple values of C (inverse of lambda) to find the best regularization strength. The Cs=10 parameter in the code below means 10 different values of C are tested for each cross-validation, and the same set of 10 C values is tested across all folds. By setting Cs=10, the algorithm will automatically select 10 values for C logarithmically spaced between a very small value (like 1e−4) and a very large value (like 1e4 or higher). You can also customize the values for C instead of using the default logarithmic spacing by using the code:

```ruby
custom_C_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]

```

However, for the current project, I will let the algorithm automatically set the values. When you set Cs=10 and cv=5, the model will perform a total of 50 iterations. The best C value from cross-validation is then used to retrain the model on the entire training set. This final model, with the best lambda applied, is then used to make predictions on the testing set. See the code below for the full process. Note that if you want to see the paramaters across the CV folds, you can use the GridSearchCV() function instead of the current function that I use. 


```ruby
from sklearn.linear_model import LogisticRegressionCV

# Perform cross-validation to find the best alpha using LogisticRegressionCV
log_reg_cv = LogisticRegressionCV(
    Cs=10,  # Number of values for C (i.e., 1/lambda) to try (default 10)
    cv=5,  # Number of cross-validation folds
    penalty='l1',  # Lasso (L1) regularization
    solver='liblinear',  # Solver that supports L1 regularization
    class_weight='balanced',
    max_iter=10000,
    random_state=42
)
log_reg_cv.fit(X_train, y_train)

# Predict and evaluate
y_pred_cv = log_reg_cv.predict(X_test)
print("Lasso Regularized Logistic Regression with Class Weight Adjustment and Cross-Validation")
print(classification_report(y_test, y_pred_cv, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_pred_cv))

# Best parameter found by cross-validation
print("Best C (inverse of regularization strength):", log_reg_cv.C_)
```

Note that I set the solver, which specifies the algorithm used to optimize the logistic regression, to 'liblinear'. This method is efficient for smaller datasets and when you specifically need L1 regularization. You can change the method to 'saga' if you have a larger dataset. The method is also appropriate for sparse data. Both 'saga' and 'liblinear' support ridge and lasso regularization. Other methods also exist, such as 'sag' (Stochastic Average Gradient Descent), which supports only L2 (Ridge) regularization and is suitable for large datasets with dense features, and 'lbfgs' and 'newton-cg', which support only L2 (Ridge) regularization and are suitable for smaller datasets and multinomial loss for multiclass classification.

Here is the output using liblinear:


<img width="737" alt="Screen Shot 2024-07-30 at 5 33 00 PM" src="https://github.com/user-attachments/assets/63f097e6-abb1-4cc1-8618-efe2f9337675">

The output shows quite promising results in terms of recall for the minority class (subscribers). The model achieved a recall of 0.73 for the minority class, meaning it correctly identified 73% of actual subscribers. This matches the company goal as they want detect as many true positives as possible. However, the precision for the minority class is low at 0.08, indicating that among the predicted subscribers, only 8% were actual subscribers. The AUC score of 0.745 suggests a moderate ability to distinguish between subscribers and non-subscribers. The best regularization strength parameter (C) found through cross-validation is approximately 21.54, optimizing the balance between bias and variance in the model. Now let's take a look at feature importance:

```ruby
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Get the coefficients of the model
coefficients = log_reg_cv.coef_.flatten()  # Flatten to ensure that the coefficients array is one-dimensional

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': coefficients
})

# Sort by absolute importance
feature_importance['Absolute Importance'] = feature_importance['Importance'].abs()
feature_importance = feature_importance.sort_values(by='Absolute Importance', ascending=False)

print("Feature Importance:")
print(feature_importance[['Feature', 'Importance']])

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
plt.show()

```

The output: 

<img width="1079" alt="Screen Shot 2024-07-30 at 7 07 08 PM" src="https://github.com/user-attachments/assets/fe7c8526-34da-44b5-9f2d-4753130ce510">


The output indicates that the most significant predictors of the subscription status are 'other_activity', with the highest positive coefficient (1.127), indicating a strong positive influence, followed by 'race_asian' (0.742), and 'login_activity' (0.636). 'Customer_service_activity' (0.473) and 'unsubscribe_reason_email' (0.433) also both contribute positively, while 'unsubscribe_reason_login' (-0.430) and 'app_update_activity' (-0.412) show significant negative influences. The results highlight that specific user activities and demographic factors play crucial roles in predicting subscription status. 

**2.Ridge regression**

Now, let's take a look at ridge regression with the weight class adjustment.

```ruby
#2. Ridge regression with weight class adjustment 

log_reg_cv = LogisticRegressionCV(
    Cs=10,  # Number of values for C (i.e., 1/lambda) to try (default 10)
    cv=5,  # Number of cross-validation folds
    penalty='l2',  # Ridge (L2) regularization
    solver='liblinear',  # Solver that supports L1 and L2 regularization
    class_weight='balanced',
    max_iter=10000,
    random_state=42
)
log_reg_cv.fit(X_train, y_train)

# Predict and evaluate
y_pred_cv = log_reg_cv.predict(X_test)
print("Ridge Regularized Logistic Regression with Class Weight Adjustment and Cross-Validation")
print(classification_report(y_test, y_pred_cv, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_pred_cv))

# Best parameter found by cross-validation
print("Best C (inverse of regularization strength):", log_reg_cv.C_)
```

The output: 

<img width="742" alt="Screen Shot 2024-07-30 at 7 15 25 PM" src="https://github.com/user-attachments/assets/1ca7972b-f0ff-4193-8bd3-e4df3d033621">

According to the output, the ridge model show a high recall for the minority class (subscribers) at 0.85, indicating that the model successfully identifies 85% of actual subscribers. However, the model shows a very low precision (0.03), indicating that only 3% of predicted subscribers are actual subscribers. The overall accuracy of the model is only 0.27, and the AUC score is 0.557, suggesting limited discriminative ability between classes. On the other hand, the Lasso model that I ran previously balances both precision and recall more effectively, demonstrating a precision of 0.08 and a recall of 0.73 for the minority class. Its overall accuracy is much higher at 0.76, with an AUC of 0.745. In other words, the lasso model has a better overall performance in distinguishing between subscribers and non-subscribers. 

You may wonder why the performance evaluation metrics change significantly when we change the regularization technique. The answer is lasso tends to produce sparser models (i.e., forcing certain coefficients to be exactly zero) and focuses on the most relevant features, which can enhance precision. Ridge includes all features, potentially introducing more noise and reducing precision. Besides, notice that the best C value (1/lambda) for the ridge model was very small (0.0001), suggesting strong regularization. This might have over-penalized the coefficients and reduced the model’s discriminative power. The best C value for lasso regression was much larger (21.54) (i.e., lower lambda), indicating less aggressive regularization. This may allow the model to retain important features while setting less important ones to zero, improving precision. 

In conclusion, changing the regularization method affects certain matrics such as the precision rate because each method handles the coefficients differently, influencing how the model discriminates between classes. 

Let's look at feature importance of the ridge regression: 

```ruby
# Feature importance

# Get the coefficients of the model
coefficients = log_reg_cv.coef_.flatten()  # Flatten to ensure that the coefficients array is one-dimensional

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': coefficients
})

# Sort by absolute importance
feature_importance['Absolute Importance'] = feature_importance['Importance'].abs()
feature_importance = feature_importance.sort_values(by='Absolute Importance', ascending=False)

print("Feature Importance:")
print(feature_importance[['Feature', 'Importance']])

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
plt.show()
```

The output: 


<img width="1115" alt="Screen Shot 2024-07-30 at 7 33 46 PM" src="https://github.com/user-attachments/assets/1cb1bbf8-fc25-4ff7-ab9d-de8903d5f170">

According to the output, the most influential feature is 'activity_count_squared', with the highest positive coefficient (8.63e-04), followed by 'days_since_signup' (1.63e-04) and 'age_squared' (-1.60e-04). Note that 'age_squared' has a negative impact. The remaining features have relatively smaller coefficients, indicating less influence on the predictions. Features such as 'activity_count' and 'session_count' have coefficients close to zero, suggesting minimal impact on the subscription status. Interestingly, some coefficients appear to be zero or near-zero, which is unusual for ridge regression. This could be due to very strong regularization applied by the model. 

In conclusion, the best model we obtained today is the Lasso regression with class weight adjustment. I hope this post has been helpful for you in understanding the foundational concepts of logistic regression from a machine learning perspective. Stay tuned for the next post, where I will discuss Gradient Boosting and AdaBoost, two of the most effective algorithms for boosting model performance.
