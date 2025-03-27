
# Usage Examples

## CLI

```bash
run-aquaagent odra_data.csv --target chlorophyll --mode interactive --model autosklearn --impute autoencoder
```

## Python

```python
from aquaagent.agent import AquaAgent

agent = AquaAgent("odra_data.csv", file_type='csv', mode='autonomous')
agent.prepare_data(target_column="chlorophyll", impute_strategy="knn")
agent.train_model(method="autosklearn")
agent.visualize()
agent.explain_model()
agent.generate_reports()
```
