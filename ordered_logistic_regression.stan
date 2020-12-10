data {
  int N; // number of observations
  int M; // number of groups
  int K; // number of response categories
  int D; // number of predictors
  int<lower=1, upper=K> y[N]; // outcomes
  row_vector[D] x[N]; // predictors
  
  int g[N]; // map observations to groups 
  

  row_vector[D] x_new[N];
}
parameters {
  ordered[K-1] theta;
  vector[D] beta;
  real a[M];
  real<lower=0, upper=10> sigma;
}
model {
  a ~ normal(0, sigma); 
  for(n in 1:N) {
    y[n] ~ ordered_logistic(x[n]* beta + a[g[n]], theta);
  }
}
generated quantities{
  int<lower=1, upper=K> y_new[N]; // outcomes
  real p_new[N];
  for(n in 1:N){
    
    y_new[n] = ordered_logistic_rng(x_new[n]* beta + a[g[n]], theta);
    p_new[n]=ordered_logistic_lpmf(y_new[n]|x_new[n]* beta + a[g[n]], theta);


  }

}