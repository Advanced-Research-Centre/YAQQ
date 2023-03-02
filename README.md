# Yet Another Quantum Quantizer

An agent that searches for novel quantum universal gate set

### Influenced by:
1. [RuliadTrotter: meta-modeling Metamathematical observers](https://community.wolfram.com/groups/-/m/t/2575951)
2. [Solovay-Kitaev theorem](https://en.wikipedia.org/wiki/Solovay%E2%80%93Kitaev_theorem)

### Development plans:
1. Compare distribution of standard (h,t,tdg) and (h,t,tdg) generated via custom U
2. If same, eliminate standard gate definitions, use only custom U
3. Bloch sphere states using hierarchical hex mesh (e.g. [H3: Uber's Hexagonal Hierarchical Spatial Index](https://github.com/uber/h3))
4. Plot "fidelity score" and "resource score"
5. Allow biases resource score, i.e. each gate in set can have a different cost (e.g. runtime on a QC)
6. "Research and justify" cost function to optimize for choosing complimentarity, based on distribution of 2 scores on all mesh points (e.g. KL-div, JS-div, cross entropy) - current choice, Jensen-Shannon distance
7. Find two U gates in an overspecified gate set (u1,u1dg,u2,u2dg) such that it beats (h,t,tdg) in either of the 2 scores
8. See if tradeoff exists between gate set size and 2 scores