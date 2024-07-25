# Logistic Regression, Sigmoid Function, the Concept of Likelihood Estimation, and More

In my previous post, I discussed linear regression from a machine learning (ML) perspective. Today, let’s delve into a different type of regression used to predict binary outcomes — logistic regression.

## Why Logistic Regression?

If you're like me, the first time you learn about logistic regression, you may wonder, "Why can’t we use linear regression to predict a binary outcome (e.g., 0, 1 or True, False)?" The answer lies in the limitations of linear regression for binary outcomes. Linear regression predictions can extend beyond the [0, 1] range, leading to unreasonable interpretations when our target outcome is strictly within this boundary. Thus, using an alternative model that works well with binary decisions is better. Logistic regression, which uses the sigmoid function, is the answer.

Before we get to the sigmoid function, there are three key terms you should familiarize yourself with in logistic regression: probability, odds, and logit: 

* **Probability** refers to the chance of an event occurring, ranging from 0 to 1, and can be interpreted as a percentage. For example, if the probability of a customer subscribing to a new streaming service is 0.6, it means there is a 60% chance the customer will subscribe.
* **Odds** are calculated by dividing the probability of an event occurring by the probability of it not occurring.
* **Logit** is the natural logarithm of the odds. See below.


<img width="601" alt="Screen Shot 2024-07-10 at 3 19 09 PM" src="https://github.com/KayChansiri/Demo_logistic_regression_ML/assets/157029107/b69425ca-1c16-40f5-94fd-858ac388df1b">

Mathematically, you can convert logit back to odds by applying the exponential function, and you can convert odds to probability by dividing the odds by 1 plus the odds:

<img width="477" alt="Screen Shot 2024-07-25 at 8 49 48 AM" src="https://github.com/user-attachments/assets/543c3f45-9923-43dc-8f20-c0a06b9d3e6f">


I know these concepts might sound confusing at first, but bear with me. I promise that by the end of this post, you'll have a clearer understanding. Let’s get back to the question I asked previously: why do we use the sigmoid function and why logistic regression for  binary outcomes? 

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

Sure, here is your text with the requested plot:

Now imagine doing this with all customers in the dataset. You will get a plot like the one below:

<img width="459" alt="Screen Shot 2024-07-25 at 9 30 29 AM" src="https://github.com/user-attachments/assets/76275306-bfc2-43fc-8197-e15972104a5e">

One challenge that we face if we end our logistic regression work here with the sigmoid function is the challenge in interpretation. Since the line does not follow a linear relationship due to the application of the sigmoid function, a one-unit increase in *X* does not correspond to a one-unit increase in *Y*. To make our interpretation easier, we convert the sigmoid function into logit, which is mathematically equivalent and simplifies the interpretation of the results. Still confused? 

To better explain, the outcome of a signmoid function is in the form of probability, ranging between 0 and 1, according to this formula: 

<img width="183" alt="Screen Shot 2024-07-25 at 9 41 16 AM" src="https://github.com/user-attachments/assets/fed0d7ad-e4b5-4467-b3e1-c3f5f322f4ed">

The logit function does the opposite: it takes a probability (a number between 0 and 1) and converts it back into a linear combination of predictors. This is done by taking the natural logarithm of the ratio of the probability of success to the probability of failure (i.e., the odds):

<img width="198" alt="Screen Shot 2024-07-25 at 9 42 39 AM" src="https://github.com/user-attachments/assets/8687d9cb-20c1-4aa7-941d-c81944da985c">

Remember I said earlie that X can be represented by your typical linear regresion function. Thus, converting a sigmoid function to logit for an easier interpretation of the output results in the following equation: 

<img width="281" alt="Screen Shot 2024-07-25 at 9 45 04 AM" src="https://github.com/user-attachments/assets/1477c19e-e560-4b1a-9a96-e95f40129563">

By converting the probability to the log-odds, we transform the nonlinear relationship into a linear one, making it easier to interpret. The coefficients in the logit model tell us how a one-unit change in a predictor affects the log-odds (i.e., logit) of the outcome.

Even though we solve the linear interpretation problem, trying to understand what ‘one unit increase in logit’ means exactly is still challenging. Thus, we often convert regression coefficients to something easier for interpretation, like odds ratios. This can be done easily by exponentiating the coefficient.

For instance, according to the age and customers' subscription I mentioned previously (the interpet = -3 and the slope = 0.1), we can say that when age is equal to zero, the odds of subscribing to the service is e<sup>-3</sup> ≈ 0.05. For every additional year of age, the odds of subscription increase by approximately e<sup>0.1</sup> ≈ 1.105, which means the odds are multiplied by 1.105, or increased by about 10.5%. 
Putting it together, when combining both the intercept and the coefficient for customer A who is 25 years old, x=−3+0.1⋅25=−0.5. The probability of subscription is: 

<img width="227" alt="Screen Shot 2024-07-25 at 10 22 19 AM" src="https://github.com/user-attachments/assets/77120fc8-57ad-4b85-81ff-c069f49af055">

The odds of subscribing to the service when the customer is 25 years old are e<sup>-0.5</sup> ≈ 0.607..

Note that when the coefficient is positive, we use e<sup>coefficient</sup> to get the odds ratio, and the output will be more than 1. When the coefficient is negative, e<sup>coefficient</sup>still gives us the odds ratio, but it indicates a decrease in odds as the output will be less than 1.

In the real world, we tend to have more than one predictor. We can write the logit formula of logistic regression as:

<img width="524" alt="Screen Shot 2024-07-25 at 10 24 18 AM" src="https://github.com/user-attachments/assets/dfbad6f9-9a24-48c7-93a3-ef20422ae7e6">
