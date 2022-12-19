The files contain the split models already implemented 
in sc2bench. Client Model should return elements in the
format `(tensors_to_measure_size,), (extra_info)`. Server
model should take in input in the format of 
`(*tensors_to_measure_size, *extra_info)` and output in 
the format `(tensors_to_measure_size,), (extra_info)`.

Current model is named model_1.py because long names can
be troublesome in imports. 