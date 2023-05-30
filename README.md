# Predicting_Flight_Delays

#Introduction

Flight delay has become a very important subject for air transportation all over the world because of the associated financial loses that the aviation industry is continuously going through.

According to data from the Bureau of Transportation Statistics (BTS) of the United Stated, over 20% of US flights were delayed during 2018, which resulted in a severe economic impact equivalent to circa 41 billion US$.

These delays not only cause inconvenience to the airlines, but also to passengers. With the increased travel time comes and increase in expenses associated to food and lodging and this results in added stress among passengers, but this doesn't account for the growing distrust towards the airlines, who also suffer from extra costs such as those associated to their crews, aircraft repositioning, increased fuel consumption while trying to reduce their elapse time, and many others that tarnish the airlines reputation and often result in the loss of demand by passengers.

The reasons for these delays vary a lot going from air congestion to weather conditions, mechanical problems, difficulties while boarding passengers, and simply the airlines inability to handle the demand given its capacity.

So what can be done as a passenger to avoid delayed flights? is it possible to know if your flight will be delayed before it comes up on the departure boards? or before you being inside the plane? The answer to these questions is maybe. By using Machine Learning (ML) Algorithms you can try to predict if your flight will be delayed in many ways. Of course, all of these different algorithms will have pitfalls and a certain degree of accuracy, and they will all depend on the data that they are fed.

In this project I will look at different ML algorithms including MLP Neural Networks to try to predict if a flight will be delayed or not before it is even announced on the departure boards. So I will not be aiming to get the highest accuracy possible, because if I wanted to do that, it would be quite easy by adding a series of features/categories that will biased the model in terms of predictive power. Examples of these are "departure delays" and "arrival delays". Think about it. If you go into a plane knowing already that there is a departure delay, chances are that your flight will be late at arrival. The same happens if you already know that the plane has an arrival delay. So this information will be looked at as part of the Exploratory Data Analysis (EDA), but will be taking out of the models with detailed explanations as why. Furthermore, I will run an algorithm with all of these features that biased the models to prove how easy it would be to get a high accuracy, but in reality not too useful because you will be already sitting on the plane.
