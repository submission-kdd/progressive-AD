Yes, this is still Python 2.

To try out the code, run python disambiguate.py with the following parameters:

surname:        e.g. Smith
firstinit:      e.g. L
result_db:      e.g. results/Smith_L.db (sqlite database for experimental results)
cfg_file:       e.g. configs/process_name.cfg (more parameters are specified, use process_name.cfg for evaluation and interact.cfg for interactive mode)
p_new:          0/1 for False/True, use the cost factor mu
d:              e.g. 1.0, the discounting parameter
random:         0/1 for False/True, use random merging ([rdm] baseline)
nbrdm:          0/1 for False/True, together with random=True, to get [nbrdm] baseline
top_k:          0 for off, 1+ for performing only top k merges, not used in our experiments*
1link:          0/1 for False/True, to use single-link clustering
similarity:     0 for probabilities, 1 for cosine similarity in the clustering similarity

There are four feature sets as databases in featDBs/. They will be used depending on what you pass as surname and firstinit. We have created feature sets for all the names in our experiments, but these require a lot of space, obviously.
In the future, this way of obtaining features will be changed to queries to an Elasticsearch instance.

You achieve the methods from our experiments as follows:

[rdm]:          0 0.0 1 0 0 0 0
[nbrdm]:        0 0.0 1 1 0 0 0
[pnew]:         1 0.0 0 0 0 0 0
[base]:         0 0.0 0 0 0 0 0
[base+pnew]:    1 0.0 0 0 0 0 0
[base+disc]:    0 1.0 0 0 0 0 0
[pnew+disc]:    1 1.0 0 0 0 0 0
[inits]:        0 0.0 0 0 0 0 0 and change to milojevic=true in the config file
[1link]:        0 1.0 0 0 0 1 0
[1link+cosim]:  0 1.0 0 0 0 1 1

If you use interactive mode, you can open the semilattice.dot file with xdot and view the changes to the graph. You might have to manually refresh the window.
________________
* This was meant for better comparability with random baselines, but with the ec* x-axis this is not required
