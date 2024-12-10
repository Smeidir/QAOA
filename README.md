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

