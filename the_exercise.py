
# Vasco M. J. Henriques 23 October 2022

# This is a small exercise inspired by the Generalized Central Limit Theorem, and the new "Expected Loss" (ES) metric that might replace the Value at Risk (VaR) metric for some market risk analytics and it follows the nice exercise posted here:
#"https://towardsdatascience.com/var-calculation-using-monte-carlo-simulations-40b2bb417a67"
#by Sanket Karve
# Like Sanket we use pandas, numpy, and the amazing levy_stable SciPy package. See contributors to the latter package here:
#
#https://github.com/scipy/scipy/tree/v1.9.3/scipy/stats/_levy_stable
#
#We follow the simplest definition for ES with no liquidity adjustments from "Explanatory note on the minimum capital requirements for market risk" from January 2019, ISBN 978-92-9259-236-3 produced by the Basel Committee on Banking Supervision (AKA d457 explanatory note).

# First import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.stats import norm
import math
from scipy.stats import levy_stable

# Here we use the 5 year set available at https://www.kaggle.com/datasets/camnugent/sandp500. We read it with pandas:
file = 'all_stocks_5yr.csv'
df = pd.read_csv(file)

# We start with the first ticker which happens to be American Airlines (AAL). Nothing particular about that ticker, it just gives us a realistic variance to play with. Using pandas:
AALsel = (df.loc[df['Name'] == 'AAL'])
s = AALsel.set_index('date')['open']
lenn = len(s)
ticker_rx = s.pct_change()        # day to day percent change

# Lets start with ploting the distribution of cumulative returns for all sequential 10 day periods of cumulative changes for this ticker, just to convince ourselves that it looks roughly Gaussian:
effective_10day = np.ndarray((lenn//10)+1) #floor division
j = 0
for i in range(1,lenn-1,10): #whole ticker, skip first value which is usually bad
        print(i)
        h10l = np.cumprod(ticker_rx[i:i+9]+1)
        print(h10l)
        effective_10day[j] = h10l[-1]
        j=j+1
plt.figure(3)
plt.hist(effective_10day,bins=30,color = 'g')
# 1% and 2.5% percentile drawn in dashed yellow lines.
plt.axvline(np.percentile(effective_10day,1), color='y', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(effective_10day,2.5), color='y', linestyle='dashed', linewidth=2)
plt.show()

# Now we want some statistics to run the Monte Carlo approach (which is useful primarily for the heavy tailed case to be addressed later)
ex_daily_rtn = np.mean(ticker_rx) #here we use the mean as E[X]
varsigma_rtn = ticker_rx.std()    #standard dev of return


# Now lets calculate some VaR and ES_T with T=10 for our single stock portfolio
Time=10 #No of days(steps or trading days in this case), 10 days base case as is the standard in many places but critically following the base case in d457. Note the observation period for the underlying standard deviation is 5 years.
histo = np.ndarray(10000)
for i in range(10000): #10000 runs of simulation
    daily_returns = (np.random.normal(ex_daily_rtn,varsigma_rtn,Time))
    h10 = np.cumprod(daily_returns+1) #for small
    histo[i] = h10[-1]
plt.figure(1)
plt.hist(histo,bins=300,color='b')
#1% and 2.5% percentile drawn in dashed yellow lines.
plt.axvline(np.percentile(histo,1), color='y', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(histo,2.5), color='y', linestyle='dashed', linewidth=2)
plt.show()

# Now, note that, in the simulation above, 100000 periods of 10 days can go by and AAL would never experience a loss higher than 30% (nor would any other ticker with similar variance and for which a Gaussian would apply, again nothing specific about AAL). So clearly the Gaussian case is suspicious as over such a long period, in reality, one would expect more remarkable periods of cumulative 10 day evolution. Alternative distributions can be found for assets in the literature but the Gaussian is of course a great "vanilla" case to play with due to the strength of the Central Limit Theorem. Sticking with the Gaussian we note that the following relation between ES and VaR, present in the footnote 8 of "d457 explanatory" holds for this case, i.e.: the ES computation at a percentile of 2.5% yields roughly the VaR at the 1% percentile, and does so here for the Gaussian case:

# VaR at 1% percentile
np.percentile(histo,1)
# ~0.85 or 15% VaR

#Expected Shortfall metric using 2.5% percentile as per the simplest definition of d457 without any liquidity adjustments:
np.mean(histo[np.where(histo < np.percentile(histo,2.5))])
# ~0.85 or 15% Expected Shortfall which is the same as VaR at 1% as footnote 8 noted


#Now lets say our observation period deceived us and that we are, in fact, in a market where the Generalized Central Limit Theorem applies and our random variable is actualy very similar to a Gaussian but slightly heavy tailed. Then we want to try a Lévy alpha-stable distribution with an alpha very close to 2. Lets do so as before but now using the amazing package levy_stable package from scipy.stats. Lets do so with an alpha of 1.99 so that we have something that looks Gaussian, especially if under sampled, but is heavy-tailed:

alpha, beta = 1.99, 0 #Gaussian would be alpha=2, Beta=0 means we have a centered distribution.
daily_returnsl = np.ndarray(10000)
Time=10 #No of days(steps or trading days in this case), 10 following Basel rules,
histol = np.ndarray(10000)
for i in range(10000): #10000 runs of simulation # scaling with 1.4*varsigma_rtn/math.sqrt(Time)
    daily_returnsl = levy_stable.rvs(alpha, beta, scale=varsigma_rtn/np.sqrt(2),size=10)+ex_daily_rtn
    h10l = np.cumprod(daily_returnsl+1) #for small
    histol[i] = h10l[-1]
plt.figure(2)
plt.hist(histol,bins=300,color='r')
#1% and 2.5% percentile drawn in dashed yellow lines.
plt.axvline(np.percentile(histol,1), color='y', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(histol,2.5), color='y', linestyle='dashed', linewidth=2)
plt.show()

# Just printing the values, first VaR at 1%
np.percentile(histol,1) # this is roughly the same as the Gaussian case (0.85 or 15% VaR) but almost by design due to our scaling of varsigma_rtn/np.sqrt(2)
np.mean(histol[np.where(histol < np.percentile(histol,1))])

#And now again the expected Shortfall metric using the 2.5% percentile as per the simplest definition of d457:
np.mean(histol[np.where(histol < np.percentile(histol,2.5))]) #Now we see a more severe expected loss beyond the 1% percentile value! This value varies a bit so try it yourself but it is tipically a few percentage points worse: ~17.5%. This is the case even if one removes wild outliers (such as a negative percentage).

# So in an heavy-tailed World, the Expected Loss metric, even in its simplest form and with no adjustments, might indeed be a better metric than VaR at capturing the scale of loss risk, and it is not replaceable by simply selecting a VaR at lower percentiles. This essentialy recaptures footnote 8 of d457 explanatory note but here we see it with two Lévy alpha-stable distributions: the ever common non heavy-tailed Gaussian/Normal distribution (alpha=2), and a perhaps an equally common in nature Lévy alpha-stable distribution with an alpha close to, but not quite, 2.

# Running this example for a multitude of assets the qualitative results are invariably the same: unlike in the Gaussian case, ES at the 2.5% percentile leads to losses that exceed the Gaussian case at 1% percentile.

# All views expressed are my own. This article is not to be treated as expert investment advise.
