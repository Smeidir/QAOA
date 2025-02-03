Next time:




Implementations:

- Combine warm-start and recursive

todo

Runs: 
- Feel for classical vs quantum -> run classic optimizer and scale WITH CONSTRAINTS
- Compare (for 2-cut first, maybe k-cut after):
    - Warm start
    - Recursive
    - Normal
    - Warm start and recursive
- Reduce the problem size so that i can see it converges on IBM
- Check fully connected graph for k-cut - when does it stop being able to run on IBM?

Report:

bACKGROUND: 
- Familiarize reader about the topic - why am i doing all this??
- Examine the state of quantum computing for the max cut problem in realtion to IBM's connectivity map?
- For example: reduce until i can get results and discuss why, springbrett into how the connectivity works
- Explaining connectivity constraints
- Is there a function that can tell me whether or not the graph can represent a given problem?
- Ideally, text size of the plots should be the same size as the text.

Q for finley:
- Is offset important when you trasnlate from qubo?

todo master:
- Find mixer operators which preserve hard constraints
- Summer school
- Look at parameter initializers on page 22 - What do i want to do?
- Check ut goemanns-williamson algorithm  - parameter estimation with this?
- If cobyla works by using the quantum circuit as an oracle, what is the criteria for the oracle to outperform an ordinary computer?
- times of multi-angle vs using normal multiple times
- do recursive until a given stop criteria of correlation is reached
- How much do we save on recursive QAOA, compared to using the extra runs for just running it longer?
- UOBYQA as optimizer
- DOes QAOA even work? with classic parameters? 
- Look into encoding multiple variables on one qubit https://arxiv.org/abs/2011.06535%20Quantum%20Random%20Access%20Codes%20for%20Boolean%20Functions


ineresting papers: 

https://arxiv.org/pdf/2209.00415 - quantum walks and MA-QAOA
https://arxiv.org/pdf/2403.00367 - quantum aco
https://arxiv.org/pdf/2302.03711 encodings for many problems, including one-hot
https://www.researchgate.net/publication/344066924_The_capacitated_MAX_k-CUT_on_a_Quantum_Computer max k-cut in quantum
https://link.springer.com/article/10.1007/s42979-020-00437-z - max k-cut in qaoa
https://quantum-journal.org/papers/q-2023-09-14-1111/pdf/ encoding toolkit and design tradeoffs
https://arxiv.org/pdf/2306.09198 bible
https://ar5iv.labs.arxiv.org/html/2112.11354#S4.SS2 using warm start inspired by goemanss-williamsen


https://arxiv.org/pdf/2202.03459Scaling%20of%20the%20quantum%20approximate%20optimization%20algorithm%20on%20superconducting%20qubit%20based%20hardware
https://arxiv.org/pdf/2006.14904Layerwise%20learning%20for%20quantum%20neural%20networks

https://www.nature.com/articles/s41467-020-144
54-2Training%20deep%20quantum%20neural%20networks

https://ieeexplore.ieee.org/document/9831167%20Quantum%20computing%20in%20power%20systems


http://www.nnw.cz/doi/2012/NNW.2012.22.019.pdf: Mathematical background of modeling ant colony optimalization with one ant being one qubit

https://arxiv.org/pdf/2403.00367: Implementation of quantum Ant Colony Optimization on graphs which are already clustered by k-means. QACO outperforms ACO (im skeptical). 

https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/1401.2142.pdf: Quantum nearest neighbours for machine learnign

https://dl.acm.org/doi/abs/10.1145/3313276.3316366: Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics

https://arxiv.org/abs/1912.04088: Grover Adaptive Search

https://arxiv.org/abs/2011.06535%20Quantum%20Random%20Access%20Codes%20for%20Boolean%20Functions
qiskit tutorial: https://github.com/qiskit-community/qiskit-optimization/blob/stable/0.6/docs/tutorials/12_quantum_random_access_optimizer.ipynb
ibm outperforming annealer: https://arxiv.org/pdf/2406.01743

depressing: https://bpb-us-e1.wpmucdn.com/sites.harvard.edu/dist/d/274/files/2024/09/2407.12768v1-1.pdf 
solve vanilla and use ideal parameters as initializzation for multiangle?