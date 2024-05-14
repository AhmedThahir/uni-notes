# Custom 

## Regression with Custom Loss Function

```python
class CustomRegressionModel(BaseEstimator):
	"""
	All variables inside the Class should end with underscore
	"""
  def __init__(self, timeout=10, maxiter=100, maxfev=100):
    self.timeout = timeout
    self.maxiter = maxiter
    self.maxfev = maxfev
    print(timeout, maxiter, maxfev)
  
	def __str__(self):
		return str(self.model)
	def __repr__(self):
		return str(self)
	
	def mse(self, pred, true, sample_weight):
		error = pred - true
		
		loss = error**2

		# median is robust to outliers than mean
		cost = np.mean(
			sample_weight * loss
		)

		return cost

	def loss(self, pred, true):
		return self.error(pred, true, self.sample_weight)
	
	def l1(self, params):
		return np.sum(np.abs(params-self.model.initial_guess))
	def l2(self, params):
		return np.sum((params-self.model.initial_guess) ** 2)
	def l3(self, params, alpha=0.5):
		return alpha * self.l1(params) + (1-alpha)*self.l2(params)

	def reg(self, params, penalty_type="l3", lambda_reg_weight = 1.0):
		"""
		lambda_reg_weight = Coefficient of regularization penalty
		"""

		if penalty_type == "l1":
			penalty = self.l1(params)
		elif penalty_type == "l2":
			penalty = self.l2(params)
		elif penalty_type == "l3":
			penalty = self.l3(params)
		else:
			raise Exception

		return lambda_reg_weight * penalty/self.sample_size

	def cost_total(self, params, X, y):
		pred = self.model.equation(X, *params)
		return self.loss(pred, true=y) #+ self.reg(params) # regularization requires standardised parameters

	def fit(self, X, y, model, method="Nelder-Mead", cost = None, sample_weight=None, alpha=0.05):
		check_X_y(X, y) #Using self.X,self.y = check_X_y(self.X,self.y) removes column names

		self.X = X
		self.y = y

		self.n_features_in_ = self.X.shape[1]

		if sample_weight is None or len(sample_weight) <= 1: # sometimes we can give scalar sample weight same for all
			self.sample_size = self.X.shape[0]
		else:
			self.sample_size = sample_weight[sample_weight > 0].shape[0]

		self.sample_weight = (
			sample_weight
			if sample_weight is not None
			else np.full(self.sample_size, 1) # set Sample_Weight as 1 by default
		)

		self.cost = (
			cost
			if cost is not None
			else self.mse
		)

		self.model = model

		params = getfullargspec(self.model.equation).args
		params = [param for param in params if param not in ['self', "x"]]
    
    self.start_time = time.time()
		
		self.optimization = o.minimize(
			self.cost_total,
			x0 = self.model.initial_guess,
			args = (self.X, self.y),
			method = method, # "L-BFGS-B", "Nelder-Mead", "SLSQP",
			constraints = [

			],
			bounds = [
				(-1, None) for param in params # variables must be positive
			],
      maxiter = self.maxiter, # default = m * 200
      maxfev = self.maxfev, # default = m * 200
		)

		self.dof = self.sample_size - self.model.k - 1 # n-k-1

		if self.dof <= 0:
			self.popt = [0 for param in params]
			st.warning("Not enough samples")
			return self
		
		success = self.optimization.success
		if success is False:
			st.warning("Did not converge!")

		self.popt = (
			self.optimization.x
		)

		self.rmse = mse(
			self.output(self.X),
			self.y,
			sample_weight = self.sample_weight,
			squared=False
		)

		cl = 1 - (alpha/2)

		if "hess_inv" in self.optimization:
			self.covx = (
				self.optimization
				.hess_inv
				.todense()
			)

			self.pcov = list(
				np.diag(
					self.rmse *
					np.sqrt(self.covx)
				)
			)

			self.popt_with_uncertainty = [
				f"""{{ \\small (
					{round_f(popt, 6)}
					±
					{round_f(stats.t.ppf(cl, self.dof) * pcov.round(2), 2)}
				)}}""" for popt, pcov in zip(self.popt, self.pcov)
			]
		else:
			self.popt_with_uncertainty = [
				f"""{{ \\small {round_f(popt, 5)} }}""" for popt in self.popt
			]

		self.model.set_fitted_coeff(*self.popt_with_uncertainty)
		
		return self
	
	def output(self, X):
		return (
			self.model
			.equation(X, *self.popt)
		)
	
	def get_se_x_cent(self, X_cent):
		return self.rmse * np.sqrt(
			(1/self.sample_size) + (X_cent.T).dot(self.covx).dot(X_cent)
		)
	def get_pred_se(self, X):
		if False: # self.covx is not None: # this seems to be abnormal. check this
			X_cent = X - self.X.mean()
			se = X_cent.apply(self.get_se_x_cent, axis = 1)
		else:
			se = self.rmse
		return se

	def predict(self, X, alpha=0.05):
		check_is_fitted(self) # Check to verify if .fit() has been called
		check_array(X) #X = check_array(X) # removes column names

		pred = (
			self.output(X)
			.astype(np.float32)
		)

		se = self.get_pred_se(X)

		cl = 1 - (alpha/2)

		ci =  stats.t.ppf(cl, self.dof) * se

		return pd.concat([pred, pred+ci, pred-ci], axis=1)
  
  def callback(self, x):
    # callback to terminate if max_sec exceeded
    elapsed = time.time() - self.start_time
    if elapsed > self.timeout:
        warnings.warn("Terminating optimization: time limit reached",
                      TookTooLong)
        return True
    else: 
        print("Elapsed: %.3f sec" % elapsed)
```

```python
model = CustomRegressionModel()
print(model) ## prints latex

model.fit(
  X_train,
  y_train,
  model = Arrhenius(),
  method = "Nelder-Mead"
)
model.predict(X_test)

print(model) ## prints latex with coefficent values
```

## Holt-Winters

```python
class HoltWinters(BaseEstimator):
    """Scikit-learn like interface for Holt-Winters method."""

    def __init__(self, season_len=24, alpha=0.5, beta=0.5, gamma=0.5):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.season_len = season_len

    def fit(self, series):
        # note that unlike scikit-learn's fit method, it doesn't learn
        # the optimal model paramters, alpha, beta, gamma instead it takes
        # whatever the value the user specified the produces the predicted time
        # series, this of course can be changed.
        beta = self.beta
        alpha = self.alpha
        gamma = self.gamma
        season_len = self.season_len
        seasonals = self._initial_seasonal(series)

        # initial values
        predictions = []
        smooth = series[0]
        trend = self._initial_trend(series)
        predictions.append(smooth)

        for i in range(1, len(series)):
            value = series[i]
            previous_smooth = smooth
            seasonal = seasonals[i % season_len]
            smooth = alpha * (value - seasonal) + (1 - alpha) * (previous_smooth + trend)
            trend = beta * (smooth - previous_smooth) + (1 - beta) * trend
            seasonals[i % season_len] = gamma * (value - smooth) + (1 - gamma) * seasonal
            predictions.append(smooth + trend + seasonals[i % season_len])

        self.trend_ = trend
        self.smooth_ = smooth
        self.seasonals_ = seasonals
        self.predictions_ = predictions
        return self
    
    def _initial_trend(self, series):
        season_len = self.season_len
        total = 0.0
        for i in range(season_len):
            total += (series[i + season_len] - series[i]) / season_len

        trend = total / season_len
        return trend

    def _initial_seasonal(self, series):
        season_len = self.season_len
        n_seasons = len(series) // season_len

        season_averages = np.zeros(n_seasons)
        for j in range(n_seasons):
            start_index = season_len * j
            end_index = start_index + season_len
            season_average = np.sum(series[start_index:end_index]) / season_len
            season_averages[j] = season_average

        seasonals = np.zeros(season_len)
        seasons = np.arange(n_seasons)
        index = seasons * season_len
        for i in range(season_len):
            seasonal = np.sum(series[index + i] - season_averages) / n_seasons
            seasonals[i] = seasonal

        return seasonals

    def predict(self, n_preds=10):
        """
        Parameters
        ----------
        n_preds: int, default 10
            Predictions horizon. e.g. If the original input time series to the .fit
            method has a length of 50, then specifying n_preds = 10, will generate
            predictions for the next 10 steps. Resulting in a prediction length of 60.
        """
        predictions = self.predictions_
        original_series_len = len(predictions)
        for i in range(original_series_len, original_series_len + n_preds):
            m = i - original_series_len + 1
            prediction = self.smooth_ + m * self.trend_ + self.seasonals_[i % self.season_len]
            predictions.append(prediction)

        return predictions
```

```python
def timeseries_cv_score(params, series, loss_function, season_len=24, n_splits=3):
    """
    Iterating over folds, train model on each fold's training set,
    forecast and calculate error on each fold's test set.
    """
    errors = []    
    alpha, beta, gamma = params
    time_series_split = TimeSeriesSplit(n_splits=n_splits) 

    for train, test in time_series_split.split(series):
        model = HoltWinters(season_len, alpha, beta, gamma)
        model.fit(series[train])

        # evaluate the prediction on the test set only
        predictions = model.predict(n_preds=len(test))
        test_predictions = predictions[-len(test):]
        test_actual = series[test]
        error = loss_function(test_actual, test_predictions)
        errors.append(error)

    return np.mean(errors)
```

```python
x = [0, 0, 0]
test_size = 20
data = series.values[:-test_size]
opt = minimize(
  timeseries_cv_score,
  x0=x, 
  args=(data, mean_squared_log_error),
  method='TNC',
  bounds=((0, 1), (0, 1), (0, 1))
)
```

```python
alpha_final, beta_final, gamma_final = opt.x

model = HoltWinters(season_len, alpha_final, beta_final, gamma_final)
model.fit(data)
predictions = model.predict(n_preds=50)

print('original series length: ', len(series))
print('prediction length: ', len(predictions))
```

