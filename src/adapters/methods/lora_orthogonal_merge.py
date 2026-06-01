"""Orthogonal subspace merging for LoRA adapters.

When several independently-trained LoRA adapters are merged by plain weighted
summation, their low-rank updates compete for the same output directions. The
overlap between the column spaces of the per-adapter ``delta_w`` matrices shows
up as cross-adapter (cross-concept) *interference*: composing the adapters
degrades the fidelity of each individual one.

This module delivers the core result of SeqLoRA — *Bilevel Orthogonal
Adaptation for Continual Multi-Concept Generation*
(https://arxiv.org/abs/2605.22743v1): learning/composing adapters whose bases
are mutually orthogonal minimises the residual interference energy and
preserves per-concept identity far better than a frozen/overlapping basis.

We do not port SeqLoRA's bilevel training loop (that needs a continual training
harness this library does not host). Instead we implement the *merge-time*
consequence of the theory: before summing the weighted ``delta_w`` matrices we
sequentially project each adapter's update onto the orthogonal complement of the
subspace already claimed by the previously processed adapters. The first adapter
keeps its full update; each subsequent one only contributes directions that do
not collide with earlier concepts. The merged update is then re-factored into
LoRA ``A``/``B`` factors of the target rank via a truncated SVD.

``residual_interference_energy`` exposes the quantity the paper bounds, so
callers can measure the interference reduction the orthogonal merge achieves.
"""

from typing import Dict, List, Sequence, Tuple

import torch


def residual_interference_energy(deltas: Sequence[torch.Tensor]) -> torch.Tensor:
    """Total cross-adapter interference energy of a set of ``delta_w`` updates.

    Defined as the sum of squared Frobenius inner products between every distinct
    pair of updates: ``sum_{i<j} <delta_i, delta_j>_F ** 2``. This is zero exactly
    when the updates are mutually orthogonal and grows as their subspaces overlap,
    matching the "residual interference energy" SeqLoRA bounds. Returned as a
    scalar tensor.
    """
    if len(deltas) < 2:
        return torch.zeros((), dtype=deltas[0].dtype if deltas else torch.float32)

    energy = torch.zeros((), dtype=torch.float32)
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            inner = torch.sum(deltas[i].to(torch.float32) * deltas[j].to(torch.float32))
            energy = energy + inner**2
    return energy


def _extend_orthonormal_basis(basis, columns: torch.Tensor, rel_tol: float):
    """Append the genuinely new directions of ``columns`` to an orthonormal ``basis``.

    Each column is orthogonalized (modified Gram-Schmidt) against the current basis
    and the directions already accepted in this call, then dropped if its residual
    norm is negligible relative to its original norm. This keeps ``basis`` exactly
    orthonormal (and therefore bounded by the row dimension), which a raw SVD
    threshold does not guarantee once near-null singular directions creep in.
    """
    for c in range(columns.shape[1]):
        v = columns[:, c].clone()
        orig_norm = torch.linalg.norm(v)
        if orig_norm <= 0:
            continue
        if basis is not None:
            v = v - basis @ (basis.transpose(-1, -2) @ v)
        residual_norm = torch.linalg.norm(v)
        if residual_norm > rel_tol * orig_norm:
            v = (v / residual_norm).unsqueeze(1)
            basis = v if basis is None else torch.cat([basis, v], dim=1)
    return basis


def orthogonalize_deltas(deltas: Sequence[torch.Tensor], rel_tol: float = 1e-5) -> List[torch.Tensor]:
    """Sequentially project each ``delta_w`` onto the orthogonal complement of the
    output column space already spanned by the preceding updates.

    The first update is returned unchanged; later updates are deflated so that
    their column space does not overlap with earlier ones. This drives the
    pairwise :func:`residual_interference_energy` of the returned updates to
    (numerically) zero while preserving as much of each adapter's novel
    contribution as the available orthogonal directions allow.
    """
    basis = None  # orthonormal basis (d x p) of the claimed output subspace
    out: List[torch.Tensor] = []
    for delta in deltas:
        d = delta.to(torch.float32)
        if basis is not None:
            # Remove the component lying in the already-claimed subspace.
            d = d - basis @ (basis.transpose(-1, -2) @ d)
        out.append(d.to(delta.dtype))

        # Extend the claimed subspace with the new (residual) directions.
        basis = _extend_orthonormal_basis(basis, d, rel_tol)
    return out


def _svd_factorize(delta_w: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """Factor a ``d x k`` update into rank-``rank`` LoRA ``A``/``B`` factors."""
    u, s, v = torch.linalg.svd(delta_w.to(torch.float32))
    u = u[:, :rank]
    s = s[:rank]
    v = v[:rank, :]
    a = v
    b = u @ torch.diag(s)
    return {"lora_A": a.to(delta_w.dtype), "lora_B": b.to(delta_w.dtype)}


def orthogonal_merge_lora(
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float],
    rank: int,
    fan_in_fan_out: bool = False,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Merge weighted LoRA ``delta_w`` matrices with orthogonal-subspace deflation.

    Args:
        deltas: Per-adapter ``delta_w`` matrices (each ``out_features x in_features``).
        weights: Mixing weight per adapter.
        rank: Target LoRA rank of the merged adapter.
        fan_in_fan_out: Whether the host layer stores weights transposed.

    Returns:
        A tuple ``(state_dict, energy_before, energy_after)`` where ``state_dict``
        holds the merged ``lora_A`` / ``lora_B`` factors and the two energies are
        the residual interference energy before and after orthogonalization (a
        direct, measurable witness of the interference the merge removed).
    """
    # Match the orientation convention used elsewhere for LoRA delta_w merging: when the host layer
    # stores weights transposed we factorize in the transposed frame and transpose the factors back.
    oriented = [torch.t(delta) if fan_in_fan_out else delta for delta in deltas]
    weighted = [w * delta for w, delta in zip(weights, oriented)]
    energy_before = residual_interference_energy(weighted)

    orthogonalized = orthogonalize_deltas(weighted)
    energy_after = residual_interference_energy(orthogonalized)

    merged = orthogonalized[0]
    for delta in orthogonalized[1:]:
        merged = merged + delta

    state_dict = _svd_factorize(merged, rank)
    if fan_in_fan_out:
        state_dict = {k: torch.t(v) for k, v in state_dict.items()}
    return state_dict, energy_before, energy_after
