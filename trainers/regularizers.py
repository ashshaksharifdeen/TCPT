import torch
from dassl.utils import Registry
from torch.nn import functional as F
import torch.nn as nn
# a mini‐registry just for regularizers
REGULARIZER_REGISTRY = Registry('regularizers')

@REGULARIZER_REGISTRY.register()
def inter_class_margin_variance(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the batch‐variance of the margin between the true‐class logit
    and the runner‐up logit for each sample.
    
    Args:
        logits: Tensor of shape (B, C) containing raw logits.
        labels: LongTensor of shape (B,) containing the ground‐truth class idx.
    
    Returns:
        A scalar Tensor: the (unbiased=False) variance of the per‐sample margins.
    """
    B, C = logits.shape
    # 1) pick out the true‐class scores
    true_scores = logits[torch.arange(B, device=logits.device), labels]  # (B,)

    # 2) mask them out so max picks the runner‐up
    logits_for_margin = logits.clone()
    logits_for_margin[torch.arange(B, device=logits.device), labels] = -float("inf")
    runner_up_scores = logits_for_margin.max(dim=1).values              # (B,)

    # 3) margin per example
    margins = true_scores - runner_up_scores                            # (B,)

    # 4) return batch‐variance of those margins
    return margins.var(unbiased=False)



@REGULARIZER_REGISTRY.register()
def margin_mean_var(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    beta:  float = 1.0,
) -> torch.Tensor:
    """
    R_margin = -alpha * mean(m_i) + beta * Var(m_i)
    where m_i = true_score - runner_up_score
    """
    B, C = logits.shape
    # true‐class scores
    true_scores = logits[torch.arange(B, device=logits.device), labels]  # (B,)
    # runner‐up scores
    tmp = logits.clone()
    tmp[torch.arange(B, device=logits.device), labels] = -float("inf")
    runner_up  = tmp.max(dim=1).values                              # (B,)
    # margins
    margins = true_scores - runner_up                               # (B,)

    mean_margin = margins.mean()
    var_margin  = margins.var(unbiased=False)
    return -alpha * mean_margin + beta * var_margin #-alpha * mean_margin 

@REGULARIZER_REGISTRY.register()
def gaussian_w2(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    2-Wasserstein batch regularizer over class-conditional Gaussians in logit space,
    using a small ridge for numeric stability.
    L_w2 = - 2/[K(K-1)] * sum_{i<j} W2^2(N(mu_i,Sigma_i), N(mu_j,Sigma_j))
    """
    device     = logits.device
    orig_dtype = logits.dtype

    # Work in float32 for stable linear algebra
    L = logits.float()
    unique = labels.unique()
    K = unique.numel()
    C = L.size(1)

    eps   = 1e-6
    eye_C = torch.eye(C, device=device, dtype=torch.float32)

    # 1) per-class mean & covariance
    means = {}
    covs  = {}
    for cls in unique.tolist():
        mask  = (labels == cls)
        Zc    = L[mask]            # [n_i, C]
        n_i   = Zc.size(0)
        mu    = Zc.mean(dim=0)     # [C]
        means[cls] = mu

        if n_i > 1:
            dev = Zc - mu          # [n_i, C]
            cov = (dev.t() @ dev) / (n_i - 1)  # [C, C]
            # symmetrize + ridge
            cov = 0.5 * (cov + cov.t()) + eps * eye_C
        else:
            # no spread with single sample
            cov = eps * eye_C
        covs[cls] = cov

    # 2) sum pairwise W2^2
    w2_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    for i in range(K):
        for j in range(i+1, K):
            ci, cj     = unique[i].item(), unique[j].item()
            mu_i, mu_j = means[ci], means[cj]
            sig_i, sig_j = covs[ci], covs[cj]

            # 2.1 mean-distance
            mean_diff2 = (mu_i - mu_j).pow(2).sum()

            # 2.2 covariance-distance via closed-form
            vals_i, vecs_i = torch.linalg.eigh(sig_i)
            sqrt_vals_i    = torch.sqrt(torch.clamp(vals_i, min=0.0))
            sig_i_sqrt     = (vecs_i * sqrt_vals_i.unsqueeze(0)) @ vecs_i.t()

            mid = sig_i_sqrt @ sig_j @ sig_i_sqrt
            # symmetrize + ridge
            mid = 0.5 * (mid + mid.T) + eps * eye_C

            vals_m, vecs_m = torch.linalg.eigh(mid)
            sqrt_vals_m    = torch.sqrt(torch.clamp(vals_m, min=0.0))
            mid_sqrt       = (vecs_m * sqrt_vals_m.unsqueeze(0)) @ vecs_m.t()

            trace_term = torch.trace(sig_i + sig_j - 2 * mid_sqrt)

            w2_sum = w2_sum + mean_diff2 + trace_term

    # 3) normalize & negate (minimize loss => maximize W2)
    if K > 1:
        coef = 2.0 / (K * (K - 1))
        L_w2 = -coef * w2_sum
    else:
        L_w2 = torch.tensor(0.0, device=device, dtype=torch.float32)

    # back to original dtype
    return L_w2.to(orig_dtype)

@REGULARIZER_REGISTRY.register()
def text_nce_align(
    text_features: torch.Tensor,
    frozen_text_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    NCE-style alignment of prompted text embeddings to frozen CLIP text embeddings.

    Args:
      text_features        Tensor[C, D]   – prompted text embeddings for all C classes
      frozen_text_features Tensor[C, D]   – frozen CLIP text embeddings for all C classes
      labels               LongTensor[B]  – ground‐truth class idx for each sample
      logit_scale          Tensor[]       – CLIP’s logit_scale (before exp)

    Returns:
      Scalar Tensor:  mean NCE loss over the batch
    """
    # 1) select z_phi for each sample: (B, D)
    z_phi = text_features[labels]                     # gather prompted-text
    # 2) normalize
    #z_phi = F.normalize(z_phi, dim=1)
    #frozen_norm = F.normalize(frozen_text_features, dim=1)

    # 3) build (B, C) logits = cos(z_phi, frozen) / tau
    tau       = logit_scale.exp()
    logits    = z_phi @ frozen_text_features.t()  # [B, C]
    #logits   /= tau

    # 4) cross-entropy against labels is exactly the NCE objective
    loss = F.cross_entropy(logits, labels)
    return loss

@REGULARIZER_REGISTRY.register()
def text_nce_align_l1(
    text_features: torch.Tensor,
    frozen_text_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    NCE‐style alignment using L1 distance, implemented via F.l1_loss:
      L = -1/B sum_i log( exp(-L1(z_phi, z0)/tau) / sum_j exp(-L1(z_phi, w0_j)/tau) )
    """

    # 1) Gather the prompted‐text embedding for each sample -> (B, D)
    z_phi = text_features[labels]  # [B, D]

    # 2) Expand to (B, C, D) so we can call F.l1_loss in one shot
    B, D = z_phi.shape
    C, _ = frozen_text_features.shape

    # (B, 1, D) -> (B, C, D)
    z_phi_exp = z_phi.unsqueeze(1).expand(-1, C, -1)    
    # (1, C, D) -> (B, C, D)
    w0_exp    = frozen_text_features.unsqueeze(0).expand(B, -1, -1)

    # 3) Elementwise L1 differences, keep per‐element => (B, C, D)
    #    Then sum over last dim to get (B, C)
    l1_per_dim = F.l1_loss(z_phi_exp, w0_exp, reduction="none")  # [B, C, D]
    dists      = l1_per_dim.sum(dim=2)                             # [B, C]

    # 4) Turn distances into logits by negation & temperature
    tau    = logit_scale.exp()  # scalar
    #logits = -dists / tau       # shape [B, C]

    # 5) Standard cross‐entropy gives exactly the NCE form
    loss = F.cross_entropy(dists, labels)
    return loss

@REGULARIZER_REGISTRY.register()
def pairwise_nce(
    tuned: torch.Tensor,
    frozen: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Batch-wise pairwise NCE between tuned[i] and frozen[i]:
      L = -1/B sum_i log
            exp(cos(tuned_i, frozen_i)/tau)
            / sum_j exp(cos(tuned_i, frozen_j)/tau)

    tuned:        (B, D) — e.g. model.imfeatures or model.textfeatures
    frozen:       (B, D) — the corresponding frozen-CLIP features
    logit_scale:  ()    — CLIP's logit_scale parameter (pre-exp)
    """
    # 1) L2-normalize
    tuned_norm  = tuned #F.normalize(tuned,  dim=1)  # (B, D)
    frozen_norm = frozen #F.normalize(frozen, dim=1)  # (B, D)

    # 2) compute (B, B) cosine logits
    tau    = logit_scale.exp()               # scalar
    logits = tuned_norm @ frozen_norm.t()     # (B, B)
    #logits = logits / tau

    # 3) targets are the diagonal (i→i)
    labels = torch.arange(tuned.shape[0], device=tuned.device)

    # 4) cross_entropy gives exactly the NCE form
    loss = F.cross_entropy(logits, labels)
    return loss

@REGULARIZER_REGISTRY.register()
def pairwise_hinge_l1(
    tuned: torch.Tensor,
    frozen: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Pairwise margin hinge on L1 distance:
      L = 1/[B*(B-1)] * sum_{i=1}^B sum_{j!=i}
            max(0, margin + d(i,i) - d(i,j))
    where d(i,j)=||tuned[i]-frozen[j]||_1.

    tuned:  (B, D)  — tuned CLIP features (e.g. text or image) for a batch
    frozen: (B, D)  — corresponding frozen-CLIP features for the same batch
    margin: float   — desired minimum gap between correct and incorrect
    """
    B, D = tuned.shape
    # 1) pairwise L1 distances: (B, B)
    #    broadcast to compute |tuned[i] - frozen[j]| sum over feature dim
    dists = (tuned.unsqueeze(1) - frozen.unsqueeze(0)).abs().sum(dim=2)  # [B, B]

    # 2) positives on the diagonal: d(i,i)
    diag_idx = torch.arange(B, device=tuned.device)
    pos = dists[diag_idx, diag_idx]  # [B]

    # 3) expand pos to (B, B) so we can compare against each column
    pos_exp = pos.unsqueeze(1).expand(-1, B)  # [B, B]

    # 4) hinge matrix: max(0, margin + pos_i - d(i,j))
    loss_mat = F.relu(margin + pos_exp - dists)  # [B, B]

    # 5) zero out the diagonal (i==j), since we don't hinge on the positive itself
    loss_mat[diag_idx, diag_idx] = 0.0

    # 6) average over all B*(B-1) off-diagonal terms
    loss = loss_mat.sum() / (B * (B - 1))
    return loss

@REGULARIZER_REGISTRY.register()
def text_covariance_match(
    text_features: torch.Tensor,
    frozen_text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Batch-wise covariance matching loss on text features:
      L = || Cov(text_features) - Cov(frozen_text_features) ||_F^2

    Args:
      text_features        Tensor[N, D] — tuned prompt embeddings
      frozen_text_features Tensor[N, D] — frozen-CLIP text embeddings
    Returns:
      scalar Tensor
    """
    # 1) center both sets
    N, D = text_features.shape

    mean_phi = text_features.mean(dim=0, keepdim=True)         # [1, D]
    mean_0   = frozen_text_features.mean(dim=0, keepdim=True)  # [1, D]

    F_phi = text_features - mean_phi      # [N, D]
    F_0   = frozen_text_features - mean_0 # [N, D]

    # 2) compute covariances: C = (F^T F) / N
    #    so shape [D, D]
    C_phi = F_phi.transpose(0,1) @ F_phi / N
    C_0   = F_0.transpose(0,1)   @ F_0   / N

    # 3) squared Frobenius norm of the difference
    loss  = (C_phi - C_0).pow(2).sum()
    return loss

@REGULARIZER_REGISTRY.register()
def text_moment_matching(
    tuned: torch.Tensor,
    frozen: torch.Tensor,
) -> torch.Tensor:
    """
    Match mean & covariance of tuned vs. frozen text features batch-wise.

    tuned:  Tensor[B, D] — tuned model’s text embeddings for each sample
    frozen: Tensor[B, D] — frozen CLIP text embeddings for each sample
    Returns:
      Scalar loss = ||mu_tuned - mu_frozen||^2 + ||cov_tuned - cov_frozen||_F^2
    """
    B, D = tuned.shape

    # 1) compute per‐batch means
    mu_t = tuned.mean(dim=0, keepdim=True)    # [1, D]
    mu_f = frozen.mean(dim=0, keepdim=True)   # [1, D]

    # 2) mean‐matching term (squared L2)
    mean_cost = F.mse_loss(mu_t, mu_f, reduction="sum")
    
    # 3) center both sets
    Ct = tuned  - mu_t  # [B, D]
    Cf = frozen - mu_f  # [B, D]

    # 4) compute covariances:  (D×D) = (D×B) @ (B×D) / B
    cov_t = Ct.t() @ Ct / B
    cov_f = Cf.t() @ Cf / B

    # 5) covariance‐matching term (Frobenius norm squared)
    cov_cost = (cov_t - cov_f).pow(2).sum()
    #cov_cost = F.l1_loss(cov_t,cov_f)
    

    return  mean_cost + cov_cost             #mean_cost + cov_cost

@REGULARIZER_REGISTRY.register()
def margin_band(
    logits: torch.Tensor,
    labels: torch.Tensor,
    delta: float = 1.0,
    eps: float   = 0.2,
    beta: float  = 0.01,
) -> torch.Tensor:
    """
    Penalize margins outside [delta - eps, delta + eps], plus optional variance term.
    """
    B, C = logits.shape
    device = logits.device

    # 1) true‐class score
    true_scores = logits[torch.arange(B, device=device), labels]           # [B]

    # 2) runner‐up score
    runner_up = logits.clone()
    runner_up[torch.arange(B, device=device), labels] = -float("inf")
    runner_up_scores = runner_up.max(dim=1).values                          

    # 3) margins
    m = true_scores - runner_up_scores   # [B]

    # 4) squared “underconfidence” penalty
    under = torch.relu((delta - eps) - m).pow(2)

    # 5) squared “overconfidence” penalty
    over  = torch.relu(m - (delta + eps)).pow(2)

    # 6) mean band‐penalty
    band_loss = (under + over).mean()

    # 7) optional variance penalty
    var_loss = m.var(unbiased=False)

    return band_loss + beta * var_loss

@REGULARIZER_REGISTRY.register()
def rafa_plus_class_repulsion(
    z_img:       torch.Tensor,    # [B, D] tuned image embeddings
    text_feats:  torch.Tensor,    # [C, D] tuned text prototypes (one per class)
    labels:      torch.LongTensor,# [B] ground‐truth class idx ∈ [0,C)
) -> torch.Tensor:
    """
    RaFA + class‐prototype repulsion (no frozen CLIP).

    - Align each z_img[i] and its matched text_feats[labels[i]]
      to the same random z_ref[i].
    - Repel z_ref[i] from *all other* class prototypes
      text_feats[j] for j != labels[i].
    """
    B, D = z_img.shape
    C, _ = text_feats.shape
    device = z_img.device

    # 1) sample one random reference per sample
    z_ref = torch.randn_like(z_img, device=device)   # [B, D]

    # 2) RaFA align to own reference
    #    we gather the matched prototype for each sample
    z_proto = text_feats[labels]                             # [B, D]
    L_rafa = F.mse_loss(z_img,   z_ref, reduction='mean') + F.mse_loss(z_proto, z_ref, reduction='mean')
   

    # 3) Repel reference from *other* class prototypes
    #    compute squared distances [B, C]
    #    expand so that each row i has dist to all class prototypes
    diffs = z_ref.unsqueeze(1) - text_feats.unsqueeze(0)     # [B, C, D]
    dist2 = diffs.pow(2).sum(dim=2)                          # [B, C]

    #    zero out the correct class so it does not repel itself
    mask = torch.ones_like(dist2, dtype=torch.bool)
    mask[torch.arange(B, device=device), labels] = False

    #    average squared‐distance over the C-1 other classes
    rep_vals = dist2[mask].view(B, C-1)                      # [B, C-1]
    L_rep    = rep_vals.mean()                               # scalar

    return 0.5 * L_rafa +  L_rep

@REGULARIZER_REGISTRY.register()
def eccv_penalty(
    zs_pred:       torch.Tensor,   
    output:  torch.Tensor,    
    
) -> torch.Tensor:
    b, c = zs_pred.shape

    min_zs, max_zs = torch.min(zs_pred,1)[0].unsqueeze(1), torch.max(zs_pred,1)[0].unsqueeze(1)
            
    constr1 = F.relu(output - max_zs.repeat(1,c)).mean()
    constr2 = F.relu(min_zs.repeat(1,c) - output).mean()                             

    return 10 * (constr1 + constr2)

@REGULARIZER_REGISTRY.register()
def eccv_zs(
    zs_pred:       torch.Tensor,   
    output:  torch.Tensor, 
    label:   torch.Tensor
    
) -> torch.Tensor:
    b, c = zs_pred.shape

    min_op, max_op = torch.min(output,1)[0].unsqueeze(1), torch.max(output,1)[0].unsqueeze(1) 
    min_zs, max_zs = torch.min(zs_pred,1)[0].unsqueeze(1), torch.max(zs_pred,1)[0].unsqueeze(1)
            
    op_norm = (output - min_op)/ (max_op - min_op)
    op_norm = op_norm * (max_zs - min_zs) + min_zs

    loss = F.cross_entropy(op_norm, label)                                

    return loss

class LogitMarginL1(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * mean( max(0, max_j l_j - l_i - margin) )

    Args:
        margin (float): the margin δ
        alpha  (float): weight on the margin term
        ignore_index (int): target value to ignore
        schedule, mu, max_alpha, step_size: optional α scheduling (not shown here)
    """
    def __init__(self,
                 margin: float = 10.0,
                 alpha: float = 0.1,
                 ignore_index: int = -100):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def get_diff(self, inputs):
        # for each sample i: compute max_j l_j - l_i
        max_vals = inputs.max(dim=1, keepdim=True).values
        return max_vals - inputs

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # flatten if needed
        if inputs.dim() > 2:
            N, C = inputs.shape[:2]
            inputs = inputs.view(N, C, -1)                  # N,C,H,W -> N,C,H*W
            inputs = inputs.permute(0,2,1).reshape(-1, C)   # -> N*H*W, C
            targets = targets.view(-1)

        # mask out ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            inputs = inputs[mask]
            targets = targets[mask]

        loss_ce     = self.cross_entropy(inputs, targets)
        diffs       = self.get_diff(inputs)                # [B, C]
        loss_margin = F.relu(diffs - self.margin).mean()   # mean over all entries
        loss_total  = loss_ce + self.alpha * loss_margin

        return loss_total, loss_ce, loss_margin


@REGULARIZER_REGISTRY.register()
def margin_l1(inputs: torch.Tensor,
              targets: torch.Tensor,
              margin: float = 10.0,
              alpha: float = 0.1) -> torch.Tensor:
    """Wrapper so you can do
           reg_fn = REGULARIZER_REGISTRY.get('margin_l1')
           loss   = reg_fn(logits, label, margin=10, alpha=0.1)
    """
    device = inputs.device
    dtype  = inputs.dtype
    # instantiate module on correct device & dtype
    module = LogitMarginL1(margin=margin, alpha=alpha) \
                 .to(device) \
                 .to(dtype)
    loss, _, _ = module(inputs, targets)
    return loss

@REGULARIZER_REGISTRY.register()
def progradloss(stu_logits: torch.Tensor,
              tea_logits: torch.Tensor,
              label: torch.Tensor,) -> torch.Tensor:
    """Wrapper so you can do
           reg_fn = REGULARIZER_REGISTRY.get('margin_l1')
           loss   = reg_fn(logits, label, margin=10, alpha=0.1)
    """
    tea_prob = F.softmax(tea_logits /1.0 , dim=-1)
    kl_loss = -tea_prob * F.log_softmax(stu_logits / 1.0,
                                            -1) * 1.0 * 1.0
    kl_loss = kl_loss.sum(1).mean()


    return kl_loss
