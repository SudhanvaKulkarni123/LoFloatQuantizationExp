# MX integration into greedy sensitivty
/plan.  Hi can you add a naive mx_search algorithm that finds the largest mx tile that either prevents all underflow or maintains model accuracy?
  The goal is to plug the algo into greedy_sensitivty, along with the exp search. 
  I first describe the seaarch for the block size, then the integration with exp search

  ## psuedocode for shape search
  - First start a greedy search across rows. start with a tile shape of (32,1). and count the number of elemnts that would underflow in both weights and activations. If the underflow amount is 0, increase the shape to (64,1). 
   - Continue till we either reach (2048,1) or we get non-zero underflow. 
   - If we reach (2048,1) return
   - Else run a forward pass with the current config (say (x,1)) and measure accuracy. 
     - If this config jas acceptable accuracy (usin the exisiting greedy sensitivty framework), then increase to (2x,1). 
     - Else return to previous config (x/2,1) and start expnading column-wise (so run test on (x/2,2). 
        - continue testing this accuracy on (x/2, 4), (x/2, 8) and so on until we hit a config where accuracy is not acceptable (evaluated by forward pass)
-  Repeat the same algorithm but over columns insetad of rows, then return whichever optimizer (over rows or cols) gives us the biggest tile (tile (a,b) haS area a*b)

 ## integration into exp search
 - The goal is to integrate the above algo into the greedy_search in sensitivty_search.py specifically in the exp search. When you reduce the number of exp bits by 1 in the greedy search after calibration, measure if there is any underflow/overflow. If yes, then run the above algorithm for each layer (on both weights and activations) to determine a good tile shape. If there is not underflow/overflow when you reduce the number of exp bits, you can just continue decreasing the number of exp bits.

 Let me know if there are any questions. All the helper functions you need can be found in Lofloat-pvt.

 