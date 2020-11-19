# TG-OUT
Source code for the TG-OUT paper - (applies to TRIAGE paper as well)

##CREATE GRAPHS##
MongoDB with a large collection of tweets, the functions included create graphs (evolution of graphs) every 5 days (depending on the tweet's datetime) - returns all such graphs

##CALCULATIONS##
A set of functions which calculate various metrics (either node attributes, degree distributions or Attribute Induced Subgraph properties)

##VISUALIZE_PLOTS##
A set of functions which produce numerous visualizations (PDF,CDF,CCDF,ODDS,regression,etc.)

##TOOLSET##
A set of functions used in other modules, e.g. get all graphs produced by the create graphs module. 

##FINAL_FUNCTIONS##
The set of functions needed to produce all the results included in the paper

### FRAUDAR _ MATRICES_CREATION ###
functions needed to calculate the fraudar metric

