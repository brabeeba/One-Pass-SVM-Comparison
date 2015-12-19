# One-Pass-SVM-Comparison
## Usage
To run the time to a epoch test, type in the following line will run 4 algorithms on various size (10^4, 10^5, 10^6, 10^7) of dataset. The time can take up to an hour. Be patient when you wait for the result.
```
python benchmark.py --t 1
```
To run the accuracy to epoch test, type in the following line with your sample size. In the paper I use 10^4, 10^5, 10^6, 10^7 as the size of the sample set.
```
python benchmark.py --t 2 --sample 100000
```

