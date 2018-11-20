This will hold the documentation for all research made on CNN, use it as a means to keep track of the work you have done
and, eventually, as something that allows people to understand what you have been up to. This will also contain all the results obtained

Use this to also keep track of all the resources that you have used to come up with your solution.

As a starting point, create a basic classifier using either the MNIST dataset or the setup:
```
 PIX2NVS ---- pixel frame ----- state of the art pixel frame classifier
                  |
                  |
              DVS data ---- classifier friendly format data ---- DVS classifier
```

Sanity checks are also essential. At more or less every stage we want to understand if our computations are correct. for example a rough sanity check for the softmax classifier loss is to compare it to -log(0.1). Obviously this example doesn't really apply to what will become our full Neural Network but you get the gist.

This will also hold how the CCN is integrated in the overall system, benefits of using one classifier over the other and most importantly, benchmarks.
Don't be scared to send personal emails to Yiannis asking questions and what not to gather a better understanding of what the expectations are here.


Different Steps that we have in mind:
1) linear classifier for starters
2) personal NN
3) SOA CNN
4) SOA Spiking neural network
--> Each neural net will obviously produce different results depending on hyperparameters so we will have to report all of the difference with regards to that and related results/Benchmarking and comparisons.



RESEARCH PAPERS USED:
https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full (some notes taken on evernote)
