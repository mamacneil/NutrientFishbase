# FishBase_Nutrient_Models.py - Code to run all nutrient models for FishBase
# Aaron MacNeil 30.04.2021

# Import python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano as T
import theano.tensor as tt
import scipy as sp
import pdb
import arviz as az

if __name__ == '__main__':
    # -------------------------------------------Helper functions------------------------------------------------------------- #

    def indexall(L):
        poo = []
        for p in L:
            if not p in poo:
                poo.append(p)
        Ix = np.array([poo.index(p) for p in L])
        return poo,Ix


    def subindexall(short,long):
        poo = []
        out = []
        for s,l in zip(short,long):
            if not l in poo:
                poo.append(l)
                out.append(s)
        return indexall(out)

    match = lambda a, b: np.array([ b.index(x) if x in b else None for x in a ])
    grep = lambda s, l: np.array([i for i in l if s in i])

    def plot_ppc_loopit(idata, title):
        fig = plt.figure(figsize=(12,9))
        ax_ppc = fig.add_subplot(211)
        ax1 = fig.add_subplot(223); ax2 = fig.add_subplot(224)
        az.plot_ppc(idata, ax=ax_ppc);
        for ax, ecdf in zip([ax1, ax2], (False, True)):
            az.plot_loo_pit(idata, y="Yi", ecdf=ecdf, ax=ax);
        ax_ppc.set_title(title)
        ax_ppc.set_xlabel("")
        return np.array([ax_ppc, ax1, ax2])

    # --------------------------------------Import data------------------------------------------------------------------ #

    # Nutrients data
    ndata = pd.read_csv('https://raw.githubusercontent.com/mamacneil/NutrientFishbase/master/data/all_nutrients_active.csv?token=AADMXIQXB2PVWE46ONK3J2LATFM4E')
    # Traits data
    tdata = pd.read_csv('https://raw.githubusercontent.com/mamacneil/NutrientFishbase/master/data/all_traits_active.csv?token=AADMXIVNKJK73H7JWLMMVS3ATFNAK')

    # --------------------------------------Merge data------------------------------------------------------------------ #

    # Add traits information to nutritional dataframe
    indx = match(ndata.spec_code.unique(),list(tdata.spec_code.values))
    rindx = match(ndata.spec_code,list(ndata.spec_code.unique()))

    # Traits to port over
    tmp = ['Class', 'Order', 'Family','Genus', 'DemersPelag','EnvTemp', 'DepthRangeDeep', 'trophic_level', 'Feeding_path', 'Lmax','BodyShape', 'K', 'tm']

    # Port over
    for trait in tmp:
        ndata[trait] = tdata[trait].values[indx][rindx]


    # --------------------------------------Full category list (in case of missing values)------------------------------ #
    # Habitat type
    Habitat = list(np.sort(pd.unique(tdata["DemersPelag"])))
    nhabs = len(Habitat)

    # Environment type
    Climate = list(np.sort(pd.unique(tdata["EnvTemp"])))
    nclim = len(Climate)

    # Feeding pathway
    FeedingPath = list(np.sort(pd.unique(tdata["Feeding_path"])))

    # Species body shape
    BodyShape = list(np.sort(pd.unique(tdata["BodyShape"])))
    nbod = len(BodyShape)

    # --------------------------------------Loop over nutrients------------------------------------------------------------------ #
    # List available nutrients
    Nutrients =  ndata.nutrient.unique()
    # Number of nutrients
    nnut = len(Nutrients)
    # Return sample sizes
    # Return sample sizes
    tmp = pd.DataFrame({'Nutrient':Nutrients,'Sample size':[sum(ndata.nutrient.values==n) for n in Nutrients]})
    tmp.to_csv('Nutrient_list.csv', index=False)
    print(tmp)

    # Model fitting function
    def HickstModel(nut,xdata):
        print('Now modelling '+nut+'...')
    
        # Subset target nutrient
        tmpdata = xdata[xdata.nutrient==nut].copy()
        # Filter out zeros?
        tmpdata = tmpdata[tmpdata.value!=0].copy()
    
        # --------------------------------------Get variables------------------------------------------------------------------ #
        # Response
        Y = tmpdata.value.values
        Ylog = np.log(Y)
    
        ## Covariates
        # Order
        Io,Order = pd.factorize(tmpdata["Order"], sort=True)
        nord = len(Order)
    
        # Habitat type
        #Ih,Habitat = pd.factorize(tmpdata["DemersPelag"], sort=True)
        Ih = np.array([Habitat.index(x) for x in tmpdata["DemersPelag"]])
    
        # Environment type
        #Ic,Climate = pd.factorize(tmpdata["EnvTemp"], sort=True)
        Ic = np.array([Climate.index(x) for x in tmpdata["EnvTemp"]])    
        # Species max depth
        MaxDepth = np.log(tmpdata['DepthRangeDeep'].values)
    
        # Trophic level
        TL = tmpdata['trophic_level'].values
    
        # Feeding pathway
        #If,FeedingPath = pd.factorize(tmpdata["Feeding_path"], sort=True)
        If = np.array([FeedingPath.index(x) for x in tmpdata["Feeding_path"]])

        # Species maximum length
        LMax = np.log(tmpdata['Lmax'].values)

        # Species body shape
        #Ib,BodyShape = pd.factorize(tmpdata["BodyShape"], sort=True)
        Ib = np.array([BodyShape.index(x) for x in tmpdata["BodyShape"]])
    
        # Growth coefficient
        K = tmpdata['K'].values
        # Age at maturity
        tm = np.log(tmpdata['tm'].values)

        ## Nussiance parameters
        # Form of sample
        Im,Form = pd.factorize(tmpdata["sample_form"], sort=True)
        nform = len(Form)
        # Preparation of sample
        Ip,Prep = pd.factorize(tmpdata["prep_form"], sort=True)
        nprep = len(Prep)


        # --------------------------------------Specify Bayesian model------------------------------------------------------------------ #
        # Labelling for ArViz
        coords = {'Order': Order,'Habitat': Habitat, 'Climate': Climate, 'FeedingPath': FeedingPath, 'BodyShape': BodyShape, 'Form': Form,
            'Prep': Prep}

        # Regularizing prior standard deviation for Normals
        Nsd = 1
    
        # Build model
        with pm.Model(coords=coords) as Model_1:
            #"""
            # Intercept
            γ0 = pm.Normal('Intercept', 0, Nsd)
            # Observation model
            σγ = pm.Exponential('Sigma_γ', 1)
            β0_ = pm.Normal('Order_nc', 0, 1, dims='Order')
            β0 = pm.Deterministic('Order_x', γ0+β0_*σγ, dims='Order')

            # Habitat type
            β1 = pm.Normal('Habitat_x', 0, Nsd, dims='Habitat')
            # Climate
            β2 = pm.Normal('Climate_x', 0, Nsd, dims='Climate')
            # Maximum Depth
            β3 = pm.Normal('MaxDepth', 0, Nsd)
            # Total Length
            β4 = pm.Normal('TL', 0, Nsd)
            # Pelagic/demersal
            β5 = pm.Normal('FeedingPath_x', 0, Nsd, dims='FeedingPath')
            # Maximum length
            β6 = pm.Normal('LMax', 0, Nsd)
            # Body form
            β7 = pm.Normal('BodyShape_x', 0, Nsd/2, dims='BodyShape')
            # Growth parameter
            β8 = pm.Normal('K', 0, Nsd)
            # Age at maturity
            β9 = pm.Normal('tm', 0, Nsd)
            # Form of sample
            β10 = pm.Normal('Form_x', 0, Nsd, dims='Form')
            # Form of prepartion
            β11 = pm.Normal('Prep_x', 0, Nsd, dims='Prep')
    
            # Mean model
            μ_ = β0[Io]+β1[Ih]+β2[Ic]+β3*MaxDepth+β4*TL+β5[If]+β6*LMax+β7[Ib]+β8*K+β9*tm+β10[Im]+β11[Ip]
    
            # Data likelihood
            if nut in ['Protein']:
                μ = μ_
                σ = pm.Uniform('Sigma', 0, 10)
                ν = pm.Uniform('nu', 0, 20)
                Yi = pm.StudentT('Yi', ν, μ, σ, observed=Ylog)
            elif nut in ['Zinc','Iron']:
                μ = pm.math.exp(μ_)
                σ = pm.Uniform('Sigma', 0, 10)
                Yi = pm.Gamma('Yi', alpha=σ, beta=σ/μ, observed=Y)
            else:
                μ = μ_
                σ = pm.Exponential('Sigma', 1)
                Yi = pm.Normal('Yi', μ, σ, observed=Ylog)
    
            ExMu = pm.Deterministic('ExMu', tt.exp(μ_))
            
            # --------------------------------------Prior predictive check------------------------------------------------------------------ #
            with Model_1:
                prior = pm.sample_prior_predictive(700)
                Model_1_priorp = az.from_pymc3(prior=prior)
            # Plot PPC
            az.plot_ppc(Model_1_priorp, group="prior").figure.savefig(nut+'_PriorPC.jpg')
        
            # --------------------------------------Sampling------------------------------------------------------------------ #
            with Model_1:
                trace = pm.sample()
            
            # --------------------------------------Export results------------------------------------------------------------------ #
            # Plot traces
            pm.plot_trace(trace).ravel()[0].figure.savefig(nut+'_Trace.jpg')
        
            # Export summary stats
            pm.summary(trace).to_csv(nut+'_Summary.csv',index=False)
        
            # Export traces
            out = pm.backends.tracetab.trace_to_dataframe(trace)
            colnames = np.array(list(out.columns), dtype=object)
            colnames[match(grep('Habitat',list(colnames)),list(colnames))] = Habitat
            colnames[match(grep('Climate',list(colnames)),list(colnames))] = Climate
            colnames[match(grep('FeedingPath',list(colnames)),list(colnames))] = FeedingPath
            colnames[match(grep('BodyShape',list(colnames)),list(colnames))] = BodyShape
            colnames[match(grep('Form',list(colnames)),list(colnames))] = Form
            colnames[match(grep('Prep',list(colnames)),list(colnames))] = Prep
            out.columns = list(colnames)
            out.to_csv(nut+'_results.csv')
        
            # --------------------------------------Posterior predictive check------------------------------------------------------------------#
            posterior_predictive = pm.sample_posterior_predictive(trace, model=Model_1)
            idata_checks = az.from_pymc3(model=Model_1, trace=trace, prior=prior, posterior_predictive=posterior_predictive)
            plot_ppc_loopit(idata_checks,nut+'_checks').ravel()[0].figure.savefig(nut+'_LooPit.jpg')
            
            # Calculate posterior predictive accuracy
            Ypred_mu = np.quantile(trace['ExMu'].T,0.5,axis=1)
            Ypred_l90 = np.quantile(trace['ExMu'].T,0.05,axis=1)
            Ypred_u90 = np.quantile(trace['ExMu'].T,0.95,axis=1)

            # Proportion of observations captured
            print('Captured '+str(np.round(np.mean(np.array([ylo<y<yup for y,ylo,yup in zip(Y,Ypred_l90,Ypred_u90)]))*100,1))+'% of '+nut+' observations')
            


#"""
if __name__ == '__main__':
    # Loop over nutrients
    for i in range(nnut):
        # Grab nutrient
        nut = Nutrients[i]
        # Fit model
        HickstModel(nut, ndata)

#"""







