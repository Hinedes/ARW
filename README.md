# ARW — Asymmetric Read/Write

**One-way gradient gate for zero-forgetting continual learning via orthogonal subspace projection.**

When a model is fine-tuned on a new domain, ARW lets it read all prior knowledge (forward pass sums all domain projectors), but only writes gradients to the new domain's own orthogonal shell (backward pass locked to one projector). The result: the base model remains immutable, while the new domain learns rapidly.

## Status
- [x] Mathematical formulation (split projectors)
- [x] GPT-2 proof-of-concept: Domain 0 retention +0.000, Domain 1 PPL 8.643
- [ ] Overnight cluster experiment 
- [ ] Jailbreak-resistance test
- [ ] Paper draft

### Related Projects

* **[Proteus][1]** — Matryoshka Subspace Freezing (first implementation of the core-freeze idea).
* **[Potential Well Project][2]** — Orthogonal subspace partitioning (precursor; identified the Orthogonality Paradox that ARW resolves).

[1]: https://github.com/hinedes/proteus
[2]: https://github.com/hinedes/potential-well-project

## Quickstart (coming soon)

```bash
git clone https://github.com/Hinedes/arw.git
cd arw
# Install dependencies
pip install torch transformers datasets
# Run ARW experiment
python train_arw.py --help
