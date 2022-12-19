This repository originates from this fork https://github.com/mazorith/sc2-benchmark-testable-splitting. 
We use [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill)
for the split and bottlenecked models. 

This is simply a server-client setup for the 
purpose of collecting stats on split models/datasets. 
Data is sent in between as a dictionary, in the form 
of `message = dict{'timestamp' + 'data'}`. 

To use, use the param_overrides to set all the correct 
variables. 

Basic Usage:

```python server.py```

```python client.py```

In the offline case, `client.py` will contain the entire
model and test the end-to-end time of the model 
as well as an evaluation of the model's performance. Evaluators
and the offline case itself will be contained in `client.py` 
only to avoid any duplication of code in `server.py`; in other
words, there is no offline evaluation for `server.py`.

Eventually, have TorchScript-ed models. 