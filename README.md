cqt
===

Two implements for Constant Q Transform

SciPy(for sparse matrix representation) and NumPy are required.

API
---
+ `fmin`: minimum frequecy;
+ `fmax`: maximum frequecy;
+ `bins`: number of bins;
+ `fs`: the rate of sample;
+ `wnd`: the window function;

Customizing window function
---
The window function `wnd` receives a parameter `N` representing the bandwith of window and returns a `numpy.array` representing the corresponding weight for each position. 


FAQ
---
Question: why does it come out `ValueError: matrices are not aligned`?
Answer: the length of sample is small. Padding some 0's to the end of the sample can fix this. 
