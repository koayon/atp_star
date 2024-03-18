# AtP*

### Improved Attribution Patching for Localizing Large Model Behaviour

This repo contains code to perform the AtP* algorithm for improved Attribution Patching. The code is based on the [AtP*: An efficient and scalable method for localizing LLM behaviour to components](https://arxiv.org/pdf/2403.00745.pdf), Kramar et al. 2024 from DeepMind.

**Attribution Patching** (AtP) was introduced in [Nanda 2022](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) as a quick approximation to the more precise _Activation Patching_ (AcP) which details the contribution of each component to some metric (e.g. NLL loss, IOI score, etc.). It works by taking the first order Taylor approximation of the contribution c(n).

## Appreciation

Thanks to Jaden and the nnsight team for the `nnsight` package that is used for the caching and interventions.

Thanks to Alice and the MechInterp Discord for the discussions and feedback.

## Usage

TODO

## Progress

- [x] Implement AtP algorithm
- [x] Implement AtP with QK-Fix algorithm improvements
- [x] Implement full AtP* with GradDrop
- [ ] Look at MLP component contributions
- [ ] Conduct ablations and throughput experiments to reproduce paper results
- [ ] Testing
- [ ] Decouple from GPT-2
- [ ] Add complete circuit-finding algorithm with subsampling and sending the highest ranked nodes to the slower AcP algorithm
- [ ] Add subsampling for diagnostic bounds
