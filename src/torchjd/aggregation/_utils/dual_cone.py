import torch
from torch import Tensor


def project_weights(U: Tensor, G: Tensor) -> Tensor:
    """
    Computes the tensor of weights corresponding to the projection of the vectors in `U` onto the
    rows of a matrix whose Gramian is provided.

    :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    """

    shape = U.shape
    m = shape[-1]
    batch_size = U.numel() // m

    # Cast to float64 for numerical stability
    G64 = G.to(dtype=torch.float64)
    U_flat = U.to(dtype=torch.float64).reshape(batch_size, m)

    W = _solve_batch_qp(U_flat, G64)

    return W.reshape(shape).to(dtype=G.dtype)


def _solve_batch_qp(U: Tensor, G: Tensor) -> Tensor:
    r"""
    Solves a batch of QPs sharing the same cost matrix using ADMM:

    .. code-block:: text

        minimize    (1/2) v^T G v
        subject to  U[i] <= v        (componentwise, for each row i of U)

    Three improvements over basic ADMM ensure convergence on ill-conditioned Gramians:

    - **Ruiz equilibration** (5 iterations): symmetrically scales G to bring all rows and
      columns to unit infinity norm, reducing the effective condition number.
    - **Adaptive rho**: the ADMM penalty parameter is updated every ``sqrt(m)`` iterations
      when primal and dual residuals are severely imbalanced, triggering a cheap re-factorization.
    - **Normalized stopping criteria**: convergence is checked against absolute + relative
      tolerances, matching the OSQP / lqp_py conventions.

    :param U: Lower-bound matrix of shape ``[B, m]``.
    :param G: Shared cost matrix of shape ``[m, m]``, symmetric positive definite.
    """

    B, m = U.shape
    device = G.device
    I_m = torch.eye(m, dtype=torch.float64, device=device)

    # --- Ruiz equilibration ---
    # Build D such that G_s = diag(D) @ G @ diag(D) has all row/column inf-norms ≈ 1.
    # Variable substitution: v_orig = D * v_scaled  =>  U_scaled = U / D.
    G_s = G.clone()
    D = torch.ones(m, dtype=torch.float64, device=device)
    for _ in range(5):
        delta = G_s.abs().amax(dim=1).clamp(min=1e-10).rsqrt()
        G_s = G_s * (delta.unsqueeze(1) * delta.unsqueeze(0))
        D = D * delta
    U_s = U / D  # [B, m] — scaled lower bounds

    # --- Rho initialization ---
    rho = max((G_s.norm("fro") / m).item(), 1e-6)

    # --- ADMM ---
    L = torch.linalg.cholesky(G_s + rho * I_m)

    V = U_s.clone()  # primal variable (scaled)
    Z = U_s.clone()  # auxiliary variable (scaled)
    u = torch.zeros(B, m, dtype=torch.float64, device=device)  # scaled dual variable

    eps_abs = eps_rel = 1e-7
    check_freq = round(m**0.5)
    tau = 10.0  # adaptive-rho trigger threshold

    for k in range(2000):
        Z_prev = Z

        # V-update: (G_s + rho*I) V = rho*(Z - u)
        V = torch.cholesky_solve((rho * (Z - u)).T, L).T

        # Z-update: project onto {z : z >= U_s}
        Z = (V + u).clamp(min=U_s)

        # Scaled dual update
        primal_residual = V - Z
        u = u + primal_residual

        if k % check_freq == 0:
            primal_res = primal_residual.norm(torch.inf).item()
            dual_res = (rho * (Z - Z_prev)).norm(torch.inf).item()

            tol_p = eps_abs + eps_rel * max(
                V.norm(torch.inf).item(),
                Z.norm(torch.inf).item(),
            )
            tol_d = eps_abs + eps_rel * (rho * u).norm(torch.inf).item()
            if primal_res < tol_p and dual_res < tol_d:
                break

            # Adaptive rho: scale rho and rescale dual variable to maintain lambda = rho * u
            if primal_res > tau * dual_res:
                rho = min(rho * tau, 1e6)
                u = u / tau
                L = torch.linalg.cholesky(G_s + rho * I_m)
            elif tau * primal_res < dual_res:
                rho = max(rho / tau, 1e-6)
                u = u * tau
                L = torch.linalg.cholesky(G_s + rho * I_m)

    if (V - Z).norm(torch.inf) > 1e-3:
        raise ValueError("Failed to solve the quadratic programming problem.")

    # Unscale: v_orig = D * v_scaled
    return V * D
