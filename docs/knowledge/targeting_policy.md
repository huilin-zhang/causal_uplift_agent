# Targeting policy

## Net value
For each customer, net value = estimated uplift x customer value - offer cost. Customer value is the margin we keep if the customer stays (here 70% of twelve monthly fees). Offer cost is the contact cost plus the discount. Contact a customer only when net value is positive.

## Budget
When the budget allows fewer offers than there are positive-net-value customers, take the highest net values first. When there are fewer positive-net-value customers than the budget, send fewer offers. Spending the full budget is not a goal.

## Why not rank by risk
Ranking by churn risk sends offers to customers who will leave anyway and to customers who would stay anyway. It also contacts sleeping dogs, whose churn rises after contact. Uplift ranking targets the customers whose decision the offer can change.

## Multiple offers
With several treatment arms, estimate uplift for each arm against control and give each customer the arm with the highest net value, or no offer if none is positive.
