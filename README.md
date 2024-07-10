# Logistic Regression, Sigmoid Function, the Concept of Likelihood Estimation, and More

In my previous post, I discussed linear regression from a machine learning (ML) perspective. Today, let’s delve into a different type of regression used to predict binary outcomes — logistic regression.

## Why Logistic Regression?

If you're like me, the first time you learn about logistic regression, you may wonder, "Why can’t we use linear regression to predict a binary outcome (e.g., 0, 1 or True, False)?" The answer lies in the limitations of linear regression for binary outcomes. Linear regression predictions can extend beyond the [0, 1] range, leading to unreasonable interpretations when our target outcome is strictly within this boundary. Thus, using an alternative model that works well with binary decisions is better. Logistic regression, which uses the sigmoid function, is the answer.

Before we get to the sigmoid function, there are three key terms you should familiarize yourself with in logistic regression: probability, odds, and logit: 

* Probability refers to the chance of an event occurring, ranging from 0 to 1, and can be interpreted as a percentage. For example, if the probability of a customer subscribing to a new streaming service is 0.6, it means there is a 60% chance the customer will subscribe.
* Odds are calculated by dividing the probability of an event occurring by the probability of it not occurring.
* Logit is the natural logarithm of the odds. See below.


<img width="601" alt="Screen Shot 2024-07-10 at 3 19 09 PM" src="https://github.com/KayChansiri/Demo_logistic_regression_ML/assets/157029107/b69425ca-1c16-40f5-94fd-858ac388df1b">

Mathematically, you can convert logit back to odds by applying the exponential function, and you can convert odds to probability by dividing the odds by 1 plus the odds:


<img width="563" alt="Screen Shot 2024-07-10 at 3 29 29 PM" src="https://github.com/KayChansiri/Demo_logistic_regression_ML/assets/157029107/12b10b88-7f03-4389-b01e-249160244a85">


I know these concepts might sound confusing at first, but bear with me. I promise that by the end of this post, you'll have a clearer understanding. Let’s get back to the question I asked previously: why do we use the sigmoid function? 


