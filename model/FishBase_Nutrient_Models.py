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
    ndata = pd.read_csv('https://raw.githubusercontent.com/mamacneil/NutrientFishbase/master/data/all_nutrients_active.csv')
    # Traits data
    tdata = pd.read_csv('https://raw.githubusercontent.com/mamacneil/NutrientFishbase/master/data/all_traits_active.csv')
    
    # --------------------------------------Merge data------------------------------------------------------------------ #

    # Add traits information to nutritional dataframe
    indx = match(ndata.spec_code.unique(),list(tdata.spec_code.values))
    rindx = match(ndata.spec_code,list(ndata.spec_code.unique()))

    # Traits to port over
    tmp = ['Class', 'Order', 'Family','Genus', 'DemersPelag','EnvTemp', 'DepthRangeDeep', 'trophic_level', 'Feeding_path', 'Lmax','BodyShape', 'K', 'tm', 'environment']

    # Port over
    for trait in tmp:
        ndata[trait] = tdata[trait].values[indx][rindx]

    # --------------------------------------Combine pelagic categories------------------------------ #
    #bathydemersal / demersal -> demersal
    ndata['DemersPelag'] = ndata['DemersPelag'].replace(['bathydemersal'],'demersal')
    #pelagic / pelagic_neritic / pelagic_oceanic -> pelagic
    ndata['DemersPelag'] = ndata['DemersPelag'].replace(['pelagic_neritic'],'pelagic')
    ndata['DemersPelag'] = ndata['DemersPelag'].replace(['pelagic_oceanic'],'pelagic')
    #benthopelagic    -> no change
    #reef_associated -> no change
    ndata['Feeding_path'] = [x+'_path' for x in ndata["Feeding_path"]]
    
    # --------------------------------------Covariates------------------------------ #
    # Habitat type
    Habitat = list(np.sort(pd.unique(ndata["DemersPelag"])))
    nhabs = len(Habitat)

    # Environment type
    Climate = list(np.sort(pd.unique(ndata["EnvTemp"])))
    nclim = len(Climate)

    # Feeding pathway
    FeedingPath = list(np.sort(pd.unique(ndata["Feeding_path"])))

    # Environment
    Environment = list(np.sort(pd.unique(ndata["environment"])))
    nenvi = len(Environment)

    # Species body shape
    BodyShape = list(np.sort(pd.unique(ndata["BodyShape"])))
    nbod = len(BodyShape)
    
    # Grab parameter mean values
    # MaxDepth
    MaxDepth_mu = np.mean(np.log(ndata['DepthRangeDeep'].values))
    # TL
    TL_mu = np.mean(ndata['trophic_level'].values)
    # Species maximum length
    LMax_mu = np.mean(np.log(ndata['Lmax'].values))
    # Growth coefficient
    K_mu = np.mean(ndata['K'].values)
    # Age at maturity
    tm_mu = np.mean(np.log(ndata['tm'].values))

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
        # Class
        Class,Icl = subindexall(tmpdata["Class"], tmpdata["Order"])
        ncl = len(Class)
        # Order
        Order,Ior = subindexall(tmpdata["Order"], tmpdata["Family"])
        nor = len(Order)
        # Family
        Family,Ifa = subindexall(tmpdata["Family"], tmpdata["Genus"])
        nga = len(Family)
        # Genus
        Genus,Ige = indexall(tmpdata["Genus"])
        ngen = len(Genus)
    
        # Habitat type
        #Ih,Habitat = pd.factorize(tmpdata["DemersPelag"], sort=True)
        Ih = np.array([Habitat.index(x) for x in tmpdata["DemersPelag"]])
    
        # Climate type
        #Ic,Climate = pd.factorize(tmpdata["EnvTemp"], sort=True)
        Ic = np.array([Climate.index(x) for x in tmpdata["EnvTemp"]])    
        
        # Environment
        Ie = np.array([Environment.index(x) for x in tmpdata["environment"]]) 
        
        # Species max depth
        MaxDepth = np.log(tmpdata['DepthRangeDeep'].values)-MaxDepth_mu
    
        # Trophic level
        TL = tmpdata['trophic_level'].values-TL_mu
    
        # Feeding pathway
        #If,FeedingPath = pd.factorize(tmpdata["Feeding_path"], sort=True)
        If = np.array([FeedingPath.index(x) for x in tmpdata["Feeding_path"]])

        # Species maximum length
        LMax = np.log(tmpdata['Lmax'].values)-LMax_mu

        # Species body shape
        #Ib,BodyShape = pd.factorize(tmpdata["BodyShape"], sort=True)
        Ib = np.array([BodyShape.index(x) for x in tmpdata["BodyShape"]])
    
        # Growth coefficient
        K = tmpdata['K'].values-K_mu
        # Age at maturity
        tm = np.log(tmpdata['tm'].values)-tm_mu

        ## Nussiance parameters
        # Form of sample
        Im,Form = pd.factorize(tmpdata["sample_form"], sort=True)
        nform = len(Form)
        # Preparation of sample
        Ip,Prep = pd.factorize(tmpdata["prep_form"], sort=True)
        nprep = len(Prep)


        # --------------------------------------Specify Bayesian model------------------------------------------------------------------ #
        # Labelling for ArViz
        coords = {'Class':Class, 'Order':Order, 'Family':Family, 'Genus':Genus, 'Climate': Climate, 'FeedingPath': FeedingPath, 'Environment':Environment,
        'BodyShape': BodyShape, 'Habitat': Habitat, 'Form': Form, 'Prep': Prep}

        # Regularizing prior standard deviation for Normals
        Nsd = 1
    
        # Build model
        with pm.Model(coords=coords) as Model_1:
            #"""
            # Intercept
            γ0 = pm.Normal('Intercept', 0, Nsd)
            # Phylogeny
            if len(coords['Class'])==1:
                # Class
                σ_o = pm.Exponential('Sigma_order', 1)
                β0_onc = pm.Normal('Ord_nc', 0, 1, dims='Order')
                β0_o = pm.Deterministic('Order_', γ0+β0_onc*σ_o, dims='Order')
            else:    
                # Class
                σ_c = pm.Exponential('Sigma_class', 1)
                β0_cnc = pm.Normal('Cla_nc', 0, 1, dims='Class')
                β0_c = pm.Deterministic('Class_', γ0+β0_cnc*σ_c, dims='Class') 

                σ_o = pm.Exponential('Sigma_order', 1)
                β0_onc = pm.Normal('Ord_nc', 0, 1, dims='Order')
                β0_o = pm.Deterministic('Order_', β0_c[Icl]+β0_onc*σ_o, dims='Order')  
    
            # Family
            σ_f = pm.Exponential('Sigma_family', 1)
            β0_fnc = pm.Normal('Fam_nc', 0, 1, dims='Family')
            β0_f = pm.Deterministic('Family_', β0_o[Ior]+β0_fnc*σ_f, dims='Family')    
    
            # Genus
            σ_g = pm.Exponential('Sigma_genus', 1)
            β0_gnc = pm.Normal('Gen_nc', 0, 1, dims='Genus')
            β0_g = pm.Deterministic('Genus_', β0_f[Ifa]+β0_gnc*σ_g, dims='Genus')

            # Habitat type
            β1 = pm.Normal('Habitat_x', 0, Nsd, dims='Habitat')
            # Climate
            β2 = pm.Normal('Climate_x', 0, Nsd, dims='Climate')
            # Maximum Depth
            #β3 = pm.Normal('MaxDepth', 0, Nsd)
            # Total Length
            β4 = pm.Normal('TL', 0, Nsd)
            # Pelagic/demersal
            β5 = pm.Normal('FeedingPath_x', 0, Nsd, dims='FeedingPath')
            # Maximum length
            β6 = pm.Normal('LMax', 0, Nsd)
            # Body form
            β7 = pm.Normal('BodyShape_x', 0, Nsd/2, dims='BodyShape')
            # Growth parameter
            #β8 = pm.Normal('K', 0, Nsd)
            # Age at maturity
            β9 = pm.Normal('tm', 0, Nsd)
            # Form of sample
            β10 = pm.Normal('Form_x', 0, Nsd, dims='Form')
            # Form of prepartion
            β11 = pm.Normal('Prep_x', 0, Nsd, dims='Prep')
            # Environment
            β12 = pm.Normal('Environment_x', 0, Nsd, dims='Environment')
    
            # Mean model
            #μ_ = β0_g[Ige]+β1[Ih]+β2[Ic]+β3*MaxDepth+β4*TL+β5[If]+β6*LMax+β7[Ib]+β8*K+β9*tm+β10[Im]+β11[Ip]
            μ_ = β0_g[Ige]+β1[Ih]+β2[Ic]+β4*TL+β5[If]+β7[Ib]+β6*LMax+β9*tm+β10[Im]+β11[Ip]+β12[Ie]
            
    
            # Data likelihood
            if nut in ['Protein']:
                μ = μ_
                ν = pm.Uniform('nu', 0, 20)
                σ = pm.Exponential('Sigma', 1)
                Yi = pm.StudentT('Yi', ν, μ, σ, observed=Y)
                ExMu = pm.Deterministic('ExMu', μ_)
            elif nut in ['Zinc','Calcium','Iron','Vitamin_A']:
                μ = μ_
                σ = pm.Exponential('Sigma', 1)
                ν = pm.Uniform('nu', 0, 20)
                Yi = pm.StudentT('Yi', ν, μ, σ, observed=Ylog)
                ExMu = pm.Deterministic('ExMu', tt.exp(μ_))
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
            tmp = pm.summary(trace)
            colnames = np.array(list(tmp.index), dtype=object)
            colnames[match(grep('Habitat',list(colnames)),list(colnames))] = Habitat
            colnames[match(grep('Climate',list(colnames)),list(colnames))] = Climate
            colnames[match(grep('FeedingPath',list(colnames)),list(colnames))] = FeedingPath
            colnames[match(grep('BodyShape',list(colnames)),list(colnames))] = BodyShape
            colnames[match(grep('Form',list(colnames)),list(colnames))] = Form
            colnames[match(grep('Prep',list(colnames)),list(colnames))] = Prep
            colnames[match(grep('Order',list(colnames)),list(colnames))] = Order
            colnames[match(grep('Family',list(colnames)),list(colnames))] = Family
            colnames[match(grep('Genus',list(colnames)),list(colnames))] = Genus
            colnames[match(grep('Environment',list(colnames)),list(colnames))] = Environment

            if len(coords['Class'])>1:
                colnames[match(grep('Class',list(colnames)),list(colnames))] = Class

            tmp.index = list(colnames)
            tmp.to_csv(nut+'_Summary.csv')
        
            # Export traces
            out = pm.backends.tracetab.trace_to_dataframe(trace)
            colnames = np.array(list(out.columns), dtype=object)
            colnames[match(grep('Habitat',list(colnames)),list(colnames))] = Habitat
            colnames[match(grep('Climate',list(colnames)),list(colnames))] = Climate
            colnames[match(grep('FeedingPath',list(colnames)),list(colnames))] = FeedingPath
            colnames[match(grep('BodyShape',list(colnames)),list(colnames))] = BodyShape
            colnames[match(grep('Form',list(colnames)),list(colnames))] = Form
            colnames[match(grep('Prep',list(colnames)),list(colnames))] = Prep
            colnames[match(grep('Order',list(colnames)),list(colnames))] = Order
            colnames[match(grep('Family',list(colnames)),list(colnames))] = Family
            colnames[match(grep('Genus',list(colnames)),list(colnames))] = Genus
            colnames[match(grep('Environment',list(colnames)),list(colnames))] = Environment

            if len(coords['Class'])>1:
                colnames[match(grep('Class',list(colnames)),list(colnames))] = Class

            out.columns = list(colnames)
            # Drop unwanted columns
            xout = list(grep('ExMu_',list(out.columns)))+list(grep('_nc_',list(out.columns)))
            out = out.drop(columns=xout)
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
            
            # Observed vs predicted plot
            fig = plt.figure(figsize=(9,9))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            # 1 to 1
            ax1.plot((Y,Y),(Ypred_l90,Ypred_u90),c='b')
            ax1.scatter(Y,Ypred_mu,s=70)
            ax1.scatter(Y,Y,s=70,c='red')
            ax1.set_ylabel('Predicted')
            ax1.set_title(nut)
            # Histogram
            ax2.hist(Y,bins=50,label='Observed')
            ax2.hist(Ypred_mu,bins=50, label='Predicted')
            ax2.set_xlabel('Observed')
            ax2.legend()
            plt.savefig(nut+'_ObsPred.jpg');
            


#"""
if __name__ == '__main__':
    # Loop over nutrients
    for i in range(nnut):
        # Grab nutrient
        nut = Nutrients[i]
        # Fit model
        HickstModel(nut, ndata)

#"""







