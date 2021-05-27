# NutrientFishbase

Modelling stemming from [Hicks **et al.** 2019](https://www.nature.com/articles/s41586-019-1592-6) to predict nutrients of species in FishBase.


**NutrientFishbase/model/FishBase_Nutrient_Models.py**:  Python code for estimating model parameters from observed nutrient data (from species in **NutrientFishbase/data/all_nutrients_active.csv** and traits from **NutrientFishbase/data/all_traits_active.csv**).

**NutrientFishbase/model/FishBase_Nutrient_Predictions.py**: Python code for using results of individual nutrient models to predict nutrient content for unobserved species (from **NutrientFishbase/data/all_traits_active.csv**), based on traits.
