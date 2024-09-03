# Architectures

## Basic

```python
class NeuralNet(nn.Module):
	def __init__(self, init_data, hidden_layers):
		super().__init__()

		for x, y in DataLoader(init_data):
			self.input_size = x.shape[-1]
			self.output_size = y.shape[-1]
			break
		
		output_layer = nn.LazyLinear(self.output_size) # output layer
		
		layers = (
	  		# [input_layer] +
			hidden_layers +
			[output_layer]
		)

		self.network = nn.Sequential(
			*layers
		)

		# init lazy layers
		self.forward(x)

	def reshape(self, x):
		# batch_size, no_of_channels, width, height
		return x.view(x.shape[0], 1, x.shape[1], x.shape[2])

	def forward(self, x):
		return self.network(self.reshape(x)).squeeze()
	
	def predict_proba(self, X):
		return self.forward(X)
	def predict_from_proba(self, proba)
		return proba.argmax(axis=1)
	def predict(self, X):
		return self.predict_from_proba(self.predict_proba(X))
```

### IDK

- $1 - k \text{ classifiers}$
- Advantage: Will save compute if lots of neurons in pre-output layer, which are connected to output layer
- Disadvantage: Looks confusing

```python
class NeuralNet(nn.Module):
	def __init__(self, init_data, hidden_layers):
		super().__init__()

		for x, y in DataLoader(init_data):
			self.input_size = x.shape[-1]
			self.output_size = y.shape[-1]
			break
		
		output_layer = nn.LazyLinear(self.output_size - 1) # output layer
		
		layers = (
	  		# [input_layer] +
			hidden_layers +
			[output_layer]
		)

		self.network = nn.Sequential(
			*layers
		)

		# init lazy layers
		self.forward(x)

	def reshape(self, x):
		# batch_size, no_of_channels, width, height
		return x.view(x.shape[0], 1, x.shape[1], x.shape[2])

	def forward(self, x):
		probs_excluding_last_class = self.network(self.reshape(x)).squeeze()
		return torch.cat(
			(
				probs_excluding_last_class,
				1 - probs_excluding_last_class.sum() # prob_last_class
			),
			dim = 1
		)
	
	def predict_proba(self, X):
		return self.forward(X)
	def predict_from_proba(self, proba)
		return proba.argmax(axis=1)
	def predict(self, X):
		return self.predict_from_proba(self.predict_proba(X))
```