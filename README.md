# Predicting_Flight_Delays

# Introduction

Flight delay has become a very important subject for air transportation all over the world because of the associated financial loses that the aviation industry is continuously going through.

According to data from the Bureau of Transportation Statistics (BTS) of the United Stated, over 20% of US flights were delayed during 2018, which resulted in a severe economic impact equivalent to circa 41 billion US$.

These delays not only cause inconvenience to the airlines, but also to passengers. With the increased travel time comes and increase in expenses associated to food and lodging and this results in added stress among passengers, but this doesn't account for the growing distrust towards the airlines, who also suffer from extra costs such as those associated to their crews, aircraft repositioning, increased fuel consumption while trying to reduce their elapse time, and many others that tarnish the airlines reputation and often result in the loss of demand by passengers.

The reasons for these delays vary a lot going from air congestion to weather conditions, mechanical problems, difficulties while boarding passengers, and simply the airlines inability to handle the demand given its capacity.

So what can be done as a passenger to avoid delayed flights? is it possible to know if your flight will be delayed before it comes up on the departure boards? or before you being inside the plane? The answer to these questions is maybe. By using Machine Learning (ML) Algorithms you can try to predict if your flight will be delayed in many ways. Of course, all of these different algorithms will have pitfalls and a certain degree of accuracy, and they will all depend on the data that they are fed.

In this project I will look at different ML algorithms including MLP Neural Networks to try to predict if a flight will be delayed or not before it is even announced on the departure boards. So I will not be aiming to get the highest accuracy possible, because if I wanted to do that, it would be quite easy by adding a series of features/categories that will biased the model in terms of predictive power. Examples of these are "departure delays" and "arrival delays". Think about it. If you go into a plane knowing already that there is a departure delay, chances are that your flight will be late at arrival. The same happens if you already know that the plane has an arrival delay. So this information will be looked at as part of the Exploratory Data Analysis (EDA), but will be taking out of the models with detailed explanations as why. Furthermore, I will run an algorithm with all of these features that biased the models to prove how easy it would be to get a high accuracy, but in reality not too useful because you will be already sitting on the plane.

# Objective

The objective of this project is very clear as described in the introduction: "Design a Model that predicts flight delays before they are announced on the departure boards"

# Data Gathering

The dataset comes from Kaggle, and it consists of a multi-year data ranging from 2009 to 2018 separated in 10 different files.

Each one of these datasets has 28 categories/features in average with a few million rows. Because of the size of each file I chose to work with only one, corresponding to the 2018. This one consists of 28 categories with just over 7.2 million rows.

Below is the glossary of all the features/categories available

Glossary

FL_DATE = Date of the Flight

OP_CARRIER = Airline Identifier

OP_CARRIER_FL_NUM = Flight Number

ORIGIN = Starting Airport Code

DEST = Destination Airport Code

CRS_DEP_TIME = Planned Departure Time

DEP_TIME = Actual Departure Time

DEP_DELAY = Total Delay on Departure in minutes

TAXI_OUT = The time duration elapsed between departure from the origin airport gate and wheels off

WHEELS_OFF = The time point that the aircraft's wheels leave the ground

WHEELS_ON = The time point that the aircraft'ss wheels touch on the ground

TAXI_IN = The time duration elapsed between wheels-on and gate arrival at the destination airport

CRS_ARR_TIME = Planned arrival time

ARR_TIME = Actual Arrival Time = ARRIVAL_TIME - SCHEDULED_ARRIVAL

ARR_DELAY = Total Delay on Arrival in minutes

CANCELLED = Flight Cancelled (1 = cancelled)

CANCELLATION_CODE = Reason for Cancellation of flight: A - Airline/Carrier; B - Weather; C - National Air System; D - Security

DIVERTED = Aircraft landed on different airport that the one scheduled

CRS_ELAPSED_TIME = Planned amount of time needed for the flight trip

ACTUAL_ELAPSED_TIME = AIR_TIME+TAXI_IN+TAXI_OUT

AIR_TIME = The time duration between wheels_off and wheels_on time

DISTANCE = Distance between two airports

CARRIER_DELAY = Delay caused by the airline in minutes

WEATHER_DELAY = Delay caused by weather

NAS_DELAY = Delay caused by air system

SECURITY_DELAY = caused by security reasons

LATE_AIRCRAFT_DELAY = Delay caused by security

Source: Kaggle

As I mentioned in the Introduction, I will be only considering features that you are aware of before the plane takes off. This way what I am predicting is before you board the plane and not while you are in the plane in mid air, which wouldn't be of much use as you would want to know if you will be late before you board the plane. Adding any of the features listed below would increase your accuracy to at least 85%, which sounds great, but then again, what's the point if you are already in the air or about to take off?

TAXI_OUT

WHEELS_OFF

WHEELS_ON

TAXI_IN

ARR_DELAY

ACTUAL_ELAPSED_TIME

Now, there is an additional feature that will biased the models, and that is the DEP_DELAY (Departure Delay), which yes, if your plane is leaving late then your chances of arriving late to your destination will increase. The plot on Figure_1, which is part of the EDA done, shows this. There I compared the DEP_DELAY with the ARR_DELAY by airline, and as you can see, normally when your flight leaves late, the airlines pushes for the flights to have shorter elapse times to compensate for the delay, and in some cases, this is accounted for and the flight ends up arriving either on time, or earlier, such as with Delta Airlines and Alaska airlines, which have both negative arrival averages, meaning an early arrival.

