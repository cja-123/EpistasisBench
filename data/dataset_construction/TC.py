import numpy as np
import pandas as pd
from jax import random
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import initialization
from tqdm import tqdm

def additive_model(df_dg,aa_len):
    # aa_len: number of amino acids (usually 20)
    # df_dg: datafame including aa1, aa2 (index), and raw deltaG values 
    
    # first/second_aa_effect: coefficients representing first/second amino acid effects [aa_len]
    first_aa_effect = numpyro.sample("first_aa_effect", dist.Normal(np.resize(0.1, (aa_len)), 3))
    second_aa_effect = numpyro.sample("second_aa_effect", dist.Normal(np.resize(0.1, (aa_len)), 3))

    # list of index of 1st aa, index of 2nd aa, [length of df_dg]
    first_aa_index = list(df_dg['aa1'])
    second_aa_index = list(df_dg['aa2'])
    
    # list of observed deltaG [length of df_dg]
    obs_dg_list = jax.numpy.array(df_dg['dG'])
    
    # pred_aa_dg_list: predicted deltaG using aa_len x2 (usually 40 = 20x2) coefficients, then clip the data bewteen -1 to 5
    pred_aa_dg_list = jax.numpy.array([first_aa_effect[index] for index in first_aa_index])+jax.numpy.array([second_aa_effect[index] for index in second_aa_index])
    clipped_pred_aa_dg_list =  jax.numpy.clip(pred_aa_dg_list,-1,5)
    
    # sigma: uncertainty of the data
    sigma = numpyro.sample("sigma", dist.Exponential(1))   
    
    # fitting coefficients by assuming predicted deltaG matches observed dG
    numpyro.sample("obs_dg",dist.TransformedDistribution(dist.Normal(0, 1), dist.transforms.AffineTransform(clipped_pred_aa_dg_list,sigma)), obs = obs_dg_list)

aas='QENHDRKTSAGMCLVIWYFP'

df_dmut = pd.read_csv('Tsuboyama2023_Dataset2_Dataset3_20230416.csv', low_memory=False)
df_dmut = df_dmut.loc[~df_dmut.pair_name.map(pd.isna)]

dmut_list = list(set(df_dmut['pair_name']))

df_sum = []

for dmut_name in tqdm(dmut_list, desc="Processing pairs"):
    sub_df = df_dmut[df_dmut['pair_name'] == dmut_name].copy()
    sub_df['aa1'] = [aas.index(x.split(':')[0][-1]) for x in sub_df['mut_type']]
    sub_df['aa2'] = [aas.index(x.split(':')[1][-1]) for x in sub_df['mut_type']]
    sub_df['dG'] = sub_df['deltaG']

    rng_key = random.PRNGKey(1)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(additive_model, init_strategy=initialization.init_to_feasible())
    mcmc = MCMC(kernel, num_warmup=100, num_samples=50, num_chains=1)

    mcmc.run(rng_key_,df_dg = sub_df,aa_len=20)
    samples_dg = mcmc.get_samples()

    first_aa_effect = np.median(samples_dg['first_aa_effect'],axis=0)
    second_aa_effect = np.median(samples_dg['second_aa_effect'],axis=0)

    sub_df['recon_dg'] = [first_aa_effect[aa1] + second_aa_effect[aa2] for aa1,aa2 in zip(sub_df['aa1'],sub_df['aa2'])]
    sub_df['thermodynamic_coupling'] = sub_df['dG'] - sub_df['recon_dg']
    df_sum.append(sub_df)

df_sum = pd.concat(df_sum)
df_sum.to_csv('thermodynamic_coupling.csv', index=False)