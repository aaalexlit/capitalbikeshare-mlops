To get the latest artifact of a certain type
eg to automatically fetch updated processed data instead of hardcoding the name

```python
import wandb
from datetime import datetime


art_type = wandb.Api().artifact_type('predictions', project='capitalbikeshare-mlops')
name_date_art = [(c.name, datetime.strptime((c._attrs.get('createdAt')), '%Y-%m-%dT%H:%M:%S')) for c in art_type.collections()]
name_date_art.sort(key=lambda nd: nd[1])
name_date_art[0][-1]

```
