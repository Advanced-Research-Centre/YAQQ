# Yet Another Quantum Quantizer

An agent that searches for novel quantum universal gate set.

It has the following motivations:
1. given a gate set, and a set of quantum circuits, find another gate set that may perform similar/better on average, but better for a subset of those circuits
2. given a set of noisy gates fabricated on a quantum processor, treat them as non-noisy and compile algorithms with that set. the noisy gates are first purified (i.e. should be unitary, but not the desired one, so it does not solve the problem of depolarizing)

### Influenced by:
1. [RuliadTrotter: meta-modeling Metamathematical observers](https://community.wolfram.com/groups/-/m/t/2575951)
2. [Solovay-Kitaev theorem](https://en.wikipedia.org/wiki/Solovay%E2%80%93Kitaev_theorem)

### How to use:
1. Install dependencies (qiskit, matplotlib, astropy, numpy, scipy)
2. Navigate to codes folder
3. >> python 8_yaqq.py

### Development plans:
1. (done) Compare distribution of standard (h,t,tdg) and (h,t,tdg) generated via custom U
2. (done) If same, eliminate standard gate definitions, use only custom U
3. (done) Plot "fidelity score" and "resource score"
4. (done) Find two U gates in an overspecified gate set (u1,u1dg,u2,u2dg) such that it beats (h,t,tdg) in either of the 2 scores
5. "Research and justify" cost function to optimize for choosing complimentarity, based on distribution of 2 scores on all mesh points (e.g. KL-div, JS-div, cross entropy) - current choice, Jensen-Shannon distance
6. See if tradeoff exists between gate set size and 2 scores
7. Bloch sphere states using hierarchical hex mesh (e.g. [H3: Uber's Hexagonal Hierarchical Spatial Index](https://github.com/uber/h3))
8. Allow biases resource score, i.e. each gate in set can have a different cost (e.g. runtime on a QC)

### Contributing:
Feel free to report issues during build or execution. We also welcome suggestions to improve the performance of this application.

### Citation:
If you find the tool useful, please consider citing:

```
@misc{YAQQ,
  author={Sarkar, Aritra and Kundu, Akash and Feld, Sebastian},
  title={YAQQ: Yet Another Quantum Quantizer},
  howpublished={\url{https://github.com/}},
  year={2023}
}
```
