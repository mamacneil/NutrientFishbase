
# Fishbase Nutrient Analysis Tool

Research has demonstrated the potential importance of fish as a critical source of micronutrients for many people, particularly among the developing world ([Vaitla **et al** 2018](https://www.nature.com/articles/s41467-018-06199-w), [Hicks **et al.** 2019](https://www.nature.com/articles/s41586-019-1592-6)).  Yet measured nutrient values are relatively scarce, with typically few species represented from only a few locations. To overcome this data limitation, we developed a Bayesian hierarchical model that includes phylogenetic information (reflecting the interrelatedness of fish species) as well as trait-based information (reflecting key aspects of fish diet) to predict concentrations of seven key nutrients for the world's fish species (calcium, iron, omega-3, protein, selenium, vitamin A, and Zinc). 

It is important to recognize that the predictions generated by our statistical model represent a set of extreme out of sample predictions - using information from less than 10% of fish species to predict the nutrient content for the remaining 90% plus species. Yet these predictions also represent our best available information about what the nutrient content of the world's fishes might be, and as such, this codebase is a work in progress that we expect to be constantly updated as new data or new covariate information becomes available. Recent fieldwork from our team has shown reasonable out of sample predictive ability from our original [Hicks **et al.** 2019](https://www.nature.com/articles/s41586-019-1592-6) for tropical fishes in Seychelles, however we also expect that the model will provide bad estimates for some species and for some locations. 

Therefore we ask that you, dear user, let us know how the model is performing against your own observations, and hope that you will be willing to contribute new data to this project via our [InFoods](http://www.fao.org/infoods/infoods/en/) collaborators, and new model structure ideas to our [model developer](mailto:a.macneil@dal.ca). These contributions will help improve the nutrient predictions available in [FishBase](https://fishbase.ca/Nutrients/NutrientSearch.php) and, ultimately, the use of fish-derived nutrients in guiding food policy around the world.

## Contents

The [NutrientFishbase repo](https://github.com/mamacneil/NutrientFishbase) on GitHub includes a few key files:

**NutrientFishbase/model/FishBase_Nutrient_Models.py**:  Python code for estimating model parameters from observed nutrient data (from species in **NutrientFishbase/data/all_nutrients_active.csv** and traits from **NutrientFishbase/data/all_traits_active.csv**).

**NutrientFishbase/model/FishBase_Nutrient_Predictions.py**: Python code for using nutrient model posteriors to predict nutrient content for unobserved species (from **NutrientFishbase/data/all_traits_for_predictions.csv**), based on phylogeny and traits.

## Use

The models include several python package dependencies, including [Pandas](https://pandas.pydata.org/) and [PyMC3](https://docs.pymc.io/).

To run the models, simply download the ``FishBase_Nutrient_Models.py`` file and run it in python

``python run FishBase_Nutrient_Models``

which will grab the required files from GitHub and generate a range of plots and files for each nutrient. Generated plots for each nutrient (X) include:

``X_LooPit.jpg``:  a three panel figure including a plot of the observed data (Yi) with their posterior predictive means, a plot of the leave-one-out probability integral transform (LOO-PIT) for the data against a uniform distribution, and a plot of the LOO-PIT expected cumulative density function (ECDF) and an expected uniform CDF, all of which look for ways in which the model is failing to fit the observed and expected data. An outline of these plots, and the source for their code, can be found [here](https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/2019/07/31/loo-pit-tutorial.html).

``X_ObsPred.jpg``:  a two-panel plot of the within-model observed (red) vs predicted (blue) values and their 95% highest posterior density intervals, and a plot of the distribution of the within-model observed and predicted values. These provide some measure of model fit and show how the models fail, generally at the highest end, suggesting additional covariates are needed to predict rare, high concentration nutrient values.

``X_PriorPC.jpg``:  a plot of the prior predictive distribution and the observed data. 

``X_Trace.jpg``:  a large figure depicting both the posterior distribution and trace for each model parameter. 

``X_results.csv``:  a flat file containing the trace for each paremeter as an individual column

``X_Summary.csv``:  a flat file with summary statistics for each parameter including the posterior mean, standard deviation (sd), lower 94% highest posterior density interval (hdi_3%), upper posterior density interval (hdi_97%), mean Monte Carlo standard error (mcse_mean), standard deviation Monte Carlo standard error (mcse_sd), effective sample size mean (ess_mean), effective sample size standard deviation (ess_sd), effective sample size central tendency (ess_bulk), effective sample size distribution tail (ess_tail), and convergence ratio (r_hat). 


To generate predictions, simply download the ``FishBase_Nutrient_Predictions.py`` file (after you've run the models) and run it in python

``python run FishBase_Nutrient_Predictions``

which will grab the ``X_results.csv`` files and will use covariates to generate posterior predictive values for all the species listed in ``all_traits_for_predictions.csv`` and generate two files:

``Species_Nutrient_Predictions.csv``:  a flat file with summary statistics for each nutrient for each species, including the scientific name of the species (species), the FishBase species code (spec_code), a highest posterior predictive density value (X_mu), a lower 95% highest posterior predictive density interval (X_l95), a lower 50% highest posterior predictive density interval (X_l50), an upper 50% highest posterior predictive density interval (X_h95), and an upper 95% highest posterior predictive density interval (X_h95).

``Species_Obs_predictions.jpg``:  a plot of the distribution of the predicted nutrients (histogram) against the range (dashed vertical lines) and median (solid vertical line) of the observed data.

## Model covariates

Fish consume nutrients in relation to key aspects of their diet, energetic demand, and thermal regime, in ways that are reflected by their individual species traits. Recognizing this, our statistical models represent these dimensions using traits sourced directly from FishBase. Specifically these include:

1. *Feeding pathway (FP)*: indicates whether nutrients are sourced through a pelagic or benthic food web.
2. *Trophic level (TL*): represents the number of feeding linkages between primary producers and a given species.
3. *Environment (EN)*: refers to the aquatic regime; one of marine, freshwater, brackish, or mixed (more than one environment).
4. *Water column (WC)*: refers to typical position in the water column; one of pelagic, demersal, reef-associated, bathypelagic, or benthopelagic, each of which has distinct pathways for nutrient input and cycling.
5. *Maximum length (Lmax)*: refers to how long a species is expected to grow, and scales directly with key attributes relating to home range size and metabolism.
6. *Age at maturity (Amat)*: reflects the time at which resources are allocated to reproduction.
7. *Body shape (BS)*:  reflects how fish feed and move through their environment; one of flat, elongate (or eel-like), fusiform, or having short-deep bodies.
8. *Geographic zone (GZ)*: represents the thermal regime typical of each species; one of tropical, subtropical, temperate, and polar/deep.

While fish traits are directly linked to where and what fish eat, these characteristics are known to be correlated among related species, resulting in phylogenetically-predictable nutrient content ([Vaitla **et al** 2018](https://www.nature.com/articles/s41467-018-06199-w)). Therefore, we  included phylogenetic relatedness within the correlation structure of our statistical model (see *Model structure* below), using a recently-developed phylogenetic tree for all marine fishes ([Rabosky et al. 2013](https://www.nature.com/articles/ncomms2958)). 

Lastly, samples of fish tissue in our nutrients database included nuisance parameters (things that influence sample collection but are not of direct interest), including the tissue type (muscle, whole, whole/parts, unkown; FO) and preparation (wet, dry, unknown; PR). 

## Model structure

Due to difficulty representing equations on GitHub, the model structure can be found [here]().

