import torch
import hdbscan
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class MoEBufferManager:
    """
    Buffer of individual "unsure" tokens used to decide when and on what to spawn
    a new expert. Before clustering, the DiT AdaLN timestep variance (shift and
    scale) is removed from the stored features so routing clusters on content
    rather than on noise level. Supports PCA + spherical k-means (and HDBSCAN) on
    the DiT's own activations, or clustering in an external encoder's space.
    """

    def __init__(self, spawn_threshold=40000, grid_h=32, grid_w=32,
                 cluster_feature="internal", cluster_method="kmeans",
                 hdbscan_min_cluster_size=250):
        self.spawn_threshold = spawn_threshold
        self.grid_h = grid_h
        self.grid_w = grid_w

        # clustering algorithm (independent of the feature space below). Assigns every token to a
        # cluster, so a large diffuse background of unsure tokens is forced into the clusters and drags 
        # the silhouette down.
        self.cluster_method = cluster_method
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size

        # "internal" (default): cluster the DiT's own token activations
        # "dino"/"clip": cluster the real images in a frozen external encoder's space instead. External features are image-level, so
        # that path clusters sample-wise (all of an image's unsure tokens shareits cluster), matching how the experts already behave. 
        self.cluster_feature = cluster_feature
        self.expert_centroids_dino = []   # coverage centroids in the external space
        self._ext_vae = None
        self._ext_encoder = None
        self._ext_vae_scale = 0.18215
        self._ext_mean = None
        self._ext_std = None
        self._ext_res = 224
        self._ext_name = "external"
        # sample = cluster whole-image embeddings (all-or-nothing per image).
        # patch  = cluster DINO patch tokens interpolated onto the latent grid, so
        #          clustering is token-wise like the internal path (an expert can
        #          own a subset of an image's tokens).
        self.cluster_granularity = "sample"
        self._ext_encoder_patch = None
        self._ext_patch_size = 14          # DINOv2 ViT patch size
        # Cap on distinct images decoded+encoded per spawn attempt (reduce cost).
        self.MAX_CLUSTER_IMAGES = 4000

        # de-duplicated image store
        self.images_latent = []      
        self.images_text = []        
        self._img_key_to_idx = {}    

        # per-token store
        self.tok_feat = []           
        self.tok_img = []            
        self.tok_pos = []            

        self.expert_centroids = []   
        # set True after the one-time seeding of initial-expert centroids
        self._seeded_initial = False

    def add_step(self, latents, text_embs, moe_module):
        # store unsure tokens into the buffer. For offline learning of the new expert the token 
        # as well as the entire image are store in the buffer
        feats = moe_module.latest_unsure_tokens
        rows = moe_module.latest_unsure_rows
        pos = moe_module.latest_unsure_pos
        if feats is None or feats.numel() == 0:
            return

        latents = latents.detach().cpu()
        text_embs = text_embs.detach().cpu()

        step_row_to_imgidx = {}
        unique_rows = torch.unique(rows)
        for r in unique_rows.tolist():
            img_idx = len(self.images_latent)
            self.images_latent.append(latents[r])
            self.images_text.append(text_embs[r])
            step_row_to_imgidx[r] = img_idx

        for j in range(feats.size(0)):
            r = int(rows[j].item())
            self.tok_feat.append(feats[j])
            self.tok_img.append(step_row_to_imgidx[r])
            self.tok_pos.append(int(pos[j].item()))

    def is_ready_to_spawn(self):
        return len(self.tok_feat) >= self.spawn_threshold

    def _pytorch_silhouette_score(self, x, labels, k):
        # same silhouette score but without using sklearn that runs on cpu
        x = x.to(torch.float32)
        N = x.size(0)
        dists = torch.cdist(x, x)
        a = torch.zeros(N, device=x.device, dtype=torch.float32)
        b = torch.full((N,), float('inf'), device=x.device, dtype=torch.float32)
        for c in range(k):
            cluster_mask = (labels == c)
            cluster_size = cluster_mask.sum()
            if cluster_size == 0:
                continue
            mean_dists_to_c = dists[:, cluster_mask].sum(dim=1) / cluster_size.float()
            if cluster_size > 1:
                in_cluster_dists = dists[cluster_mask][:, cluster_mask].sum(dim=1)
                a[cluster_mask] = in_cluster_dists / (cluster_size.float() - 1.0)
            else:
                a[cluster_mask] = 0.0
            out_cluster_mask = ~cluster_mask
            b[out_cluster_mask] = torch.minimum(b[out_cluster_mask], mean_dists_to_c[out_cluster_mask])
        sil = torch.zeros(N, device=x.device, dtype=torch.float32)
        valid = (labels.bincount()[labels] > 1)
        sil[valid] = (b[valid] - a[valid]) / torch.maximum(a[valid], b[valid]).clamp(min=1e-8)
        return sil.mean().item()

    def _spherical_kmeans(self, x, k, iters=100):
        # Spherical k-means: centroids are L2-normalized after every update. 
        # Because the inputs x are already L2-normalized, Euclidean distance maps to cosine distance.
        idx = torch.randperm(x.size(0), device=x.device)[:k]
        centroids = x[idx]
        for _ in range(iters):
            d = torch.cdist(x, centroids)
            labels = torch.argmin(d, dim=1)
            new_c = []
            for c in range(k):
                mask = (labels == c)
                if mask.sum() > 0:
                    c_mean = x[mask].mean(dim=0)
                else:
                    c_mean = x[torch.randint(0, x.size(0), (1,))].squeeze(0)
                new_c.append(c_mean)
            
            new_c = torch.stack(new_c)
            # enforce the spherical constraint
            new_c = F.normalize(new_c, p=2, dim=1, eps=1e-8)
            
            if torch.allclose(centroids, new_c, rtol=1e-4):
                break
            centroids = new_c
        return labels, centroids

    def _label_tokens(self, feat_pca_norm, sub, device, max_clusters,
                      min_silhouette, silhouette_sample):
        """
        Assign a cluster label to every token in feat_pca_norm.
        """
        method = getattr(self, "cluster_method", "kmeans")
        fsub = feat_pca_norm[sub]

        if method == "kmeans":
            sil_n = min(fsub.size(0), silhouette_sample)
            sil_idx = torch.randperm(fsub.size(0), device=device)[:sil_n]
            fsil = fsub[sil_idx]
            best_k, best_score, best_centroids = 2, -1.0, None
            for k in range(2, max_clusters + 1):
                _, cents_k = self._spherical_kmeans(fsub, k)
                sil_labels = torch.argmin(torch.cdist(fsil, cents_k), dim=1)
                score = self._pytorch_silhouette_score(fsil, sil_labels, k)
                print(f" -> Evaluated k={k} | Silhouette Score: {score:.4f}", flush=True)
                if score > best_score:
                    best_score, best_k, best_centroids = score, k, cents_k
            labels = torch.argmin(torch.cdist(feat_pca_norm, best_centroids), dim=1)
            print(f"[Clustering] Optimal k={best_k} (sil={best_score:.4f}).", flush=True)
            if best_score < min_silhouette:
                print(f"[Clustering] FAILED: best silhouette {best_score:.4f} < "
                      f"floor {min_silhouette}. No coherent structure.", flush=True)
                return None, 0, False
            return labels, best_k, True

        if method == "hdbscan":
            mcs = int(getattr(self, "hdbscan_min_cluster_size", 250))
            x_fit = fsub.detach().cpu().numpy()
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
            fit_labels = clusterer.fit_predict(x_fit)
            uniq = sorted(int(c) for c in set(fit_labels.tolist()) if c >= 0)
            n_noise = int((fit_labels < 0).sum())
            if len(uniq) == 0:
                print(f"[Clustering] FAILED: hdbscan (min_cluster_size={mcs}) found ",
                      flush=True)
                return None, 0, False

            # take sumbsample of the points and average them to get teh cluster core
            remap = {old: new for new, old in enumerate(uniq)}
            cores = []
            for old in uniq:
                pts = fsub[torch.tensor(fit_labels == old, device=device)]
                cores.append(F.normalize(pts.mean(0), p=2, dim=0, eps=1e-8))
            cores = torch.stack(cores)                       # [G, dim]

            # noise radius: assign a token to a core only if it is closer than the
            # 95th-percentile within-core distance, otherwise leave it background
            d_fit = torch.cdist(fsub, cores)
            nearest = torch.argmin(d_fit, dim=1)
            nd = d_fit.gather(1, nearest[:, None]).squeeze(1)
            radius = torch.quantile(nd.float(), 0.95).item()

            d_all = torch.cdist(feat_pca_norm, cores)
            near_all = torch.argmin(d_all, dim=1)
            nd_all = d_all.gather(1, near_all[:, None]).squeeze(1)
            labels = near_all.clone()
            labels[nd_all > radius] = -1                     # background -> noise

            G = len(uniq)
            n_assigned = int((labels >= 0).sum())
            print(f"[Clustering] hdbscan(min_cluster_size={mcs}): {G} cores, "
                  f"{n_assigned}/{labels.numel()} tokens assigned, "
                  f"{labels.numel() - n_assigned} left as background.", flush=True)
            for new, old in enumerate(uniq):
                print(f"   core {new}: {int((labels == new).sum())} tokens assigned",
                      flush=True)
            return labels, G, True

        raise ValueError(f"unknown cluster_method '{method}' (use kmeans|hdbscan)")


    # External-encoder clustering (DINO / CLIP)
    def attach_external_encoder(self, vae, encoder_fn, image_mean, image_std,
                                input_res=224, name="dino", encoder_patch_fn=None,
                                patch_size=14, granularity="sample"):
        """
        Wire in a frozen encoder + a VAE to decode buffered latents to images.

        encoder_fn:       callable(pixel_values[B,3,H,W]) -> [B, D]      (image-level)
        encoder_patch_fn: callable(pixel_values[B,3,H,W]) -> [B, S, D]   (all tokens;
                          the buffer slices the last H/ps * W/ps as patch tokens)
        Called from train_phase1 when --cluster_feature != internal.
        """
        self._ext_vae = vae.eval()
        self._ext_vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
        self._ext_encoder = encoder_fn
        self._ext_encoder_patch = encoder_patch_fn
        self._ext_patch_size = int(patch_size)
        self.cluster_granularity = granularity
        self._ext_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
        self._ext_std = torch.tensor(image_std).view(1, 3, 1, 1)
        self._ext_res = int(input_res)
        self._ext_name = name

    @torch.no_grad()
    def _embed_images_external(self, img_indices, device, chunk=32):
        """Decode buffered VAE latents back to pixels, then embed with the encoder.
        Returns [len(img_indices), D], L2-normalized, in img_indices order."""
        mean = self._ext_mean.to(device)
        std = self._ext_std.to(device)
        feats = []
        for s in range(0, len(img_indices), chunk):
            idxs = img_indices[s:s + chunk]
            lat = torch.stack([self.images_latent[i] for i in idxs]).to(device).float()
            img = self._ext_vae.decode(lat / self._ext_vae_scale).sample   # [-1,1]
            img = (img.clamp(-1, 1) + 1.0) / 2.0                            # [0,1]
            if img.size(1) == 1:                                           # gray -> 3ch
                img = img.repeat(1, 3, 1, 1)
            img = F.interpolate(img, size=(self._ext_res, self._ext_res),
                                mode="bilinear", align_corners=False)
            img = (img - mean) / std
            emb = self._ext_encoder(img.to(next(self._ext_vae.parameters()).dtype))
            feats.append(F.normalize(emb.float(), p=2, dim=1))
        return torch.cat(feats, 0)

    def _get_clustered_dataset_external(self, min_tokens_per_cluster,
                                        min_contributing_images, min_tokens_per_image,
                                        max_clusters, device, redundancy_cos_threshold,
                                        min_silhouette, silhouette_sample=4000):
        """Sample-wise clustering in the external (DINO/CLIP) space.

        Returns the same 4-tuple as the internal path (latents, texts, masks,
        centroid) so train_new_expert_offline is unchanged. The returned centroid
        is in the DiT-internal space

        The initial always-on experts (0,1) are never spawned, so they have no
        external centroid (the first external spawn has no coverage of them).
        """
        Ntok = len(self.tok_feat)
        tok_img = torch.tensor(self.tok_img)
        tok_pos = torch.tensor(self.tok_pos)
        img_ids = sorted(set(self.tok_img))
        M = len(img_ids)
        tag = f"Clustering/{self._ext_name}"
        # Decoding + encoding every distinct image (~19k) each attempt is the main
        # time sink. Cluster on a random capped subset instead.
        if M > self.MAX_CLUSTER_IMAGES:
            keep = torch.randperm(M)[: self.MAX_CLUSTER_IMAGES].tolist()
            img_ids = [img_ids[i] for i in keep]
            print(f"\n[{tag}] {M} distinct images -> sampled {len(img_ids)}; "
                  f"{Ntok} tokens buffered...", flush=True)
        else:
            print(f"\n[{tag}] {M} distinct images, {Ntok} tokens buffered...", flush=True)
        M = len(img_ids)
        if M < max(min_contributing_images, max_clusters):
            print(f"[{tag}] FAILED: only {M} distinct images.", flush=True)
            return None, None, None, None

        feats = self._embed_images_external(img_ids, device)               # [M, D]

        # deflation against already-covered external directions (empty on 1st spawn)
        if len(self.expert_centroids_dino) > 0:
            existing = F.normalize(torch.stack(self.expert_centroids_dino).to(device).float(),
                                   p=2, dim=1)
            _, S, Vh = torch.linalg.svd(existing, full_matrices=False)
            basis = Vh[S > S[0] * 1e-3]
            feats = F.normalize(feats - (feats @ basis.t()) @ basis, p=2, dim=1)

        # k by silhouette (subsampled), same spherical k-means as the internal path
        best_k, best_score, best_c = 2, -1.0, None
        for k in range(2, min(max_clusters, M) + 1):
            labels_k, cents_k = self._spherical_kmeans(feats, k)
            sn = min(M, silhouette_sample)
            sidx = torch.randperm(M, device=device)[:sn]
            score = self._pytorch_silhouette_score(feats[sidx], labels_k[sidx], k)
            print(f" -> k={k} | Silhouette {score:.4f}", flush=True)
            if score > best_score:
                best_score, best_k, best_c = score, k, cents_k
        labels = torch.argmin(torch.cdist(feats, best_c), dim=1)
        print(f"[{tag}] optimal k={best_k} (sil={best_score:.4f}).", flush=True)
        if best_score < min_silhouette:
            print(f"[{tag}] FAILED: silhouette {best_score:.4f} < {min_silhouette}.", flush=True)
            return None, None, None, None

        existing = (F.normalize(torch.stack(self.expert_centroids_dino).to(device).float(),
                                p=2, dim=1) if len(self.expert_centroids_dino) > 0 else None)

        candidates = []
        for c in range(best_k):
            rows = (labels == c).nonzero(as_tuple=True)[0]
            imgs_c = [img_ids[r] for r in rows.tolist()]
            n_img = len(imgs_c)
            if n_img < min_contributing_images:
                continue
            tokmask = torch.isin(tok_img, torch.tensor(imgs_c))
            n_tok = int(tokmask.sum().item())
            if n_tok < min_tokens_per_cluster:
                continue
            cent = F.normalize(feats[rows].mean(0), p=2, dim=0)
            novelty = (existing @ cent).max().item() if existing is not None else -1.0
            if existing is not None and novelty > redundancy_cos_threshold:
                print(f"   cluster {c}: {n_tok} tok / {n_img} img, max-cos={novelty:.3f} "
                      f"-> SKIP (covered).", flush=True)
                continue
            candidates.append((novelty, n_tok, n_img, c, imgs_c, cent))
            print(f"   cluster {c}: {n_tok} tok / {n_img} img, max-cos={novelty:.3f} "
                  f"-> candidate.", flush=True)

        if not candidates:
            print(f"[{tag}] FAILED: no uncovered image-cluster passes the gates.", flush=True)
            return None, None, None, None

        candidates.sort(key=lambda t: t[1], reverse=True)   # largest by token count
        novelty, n_tok, n_img, cid, imgs_c, cent_dino = candidates[0]
        print(f"[{tag}] selected cluster {cid}: {n_tok} tokens / {n_img} images.", flush=True)
        self.expert_centroids_dino.append(cent_dino.detach().cpu())

        # internal-space centroid (kept for router/get_expert_input_centroid consistency)
        feat_raw = torch.stack(self.tok_feat).float()
        sel = torch.isin(tok_img, torch.tensor(imgs_c))
        internal_cent = F.normalize(
            (feat_raw[sel] - feat_raw.mean(0, keepdim=True)).mean(0), p=2, dim=0)

        # masks: all buffered positions of each selected image (sample-wise)
        T = self.grid_h * self.grid_w
        out_lat, out_txt, out_mask = [], [], []
        for img in imgs_c:
            positions = tok_pos[tok_img == img]
            if positions.numel() < min_tokens_per_image:
                continue
            m = torch.zeros(T, dtype=torch.bool)
            m[positions] = True
            out_lat.append(self.images_latent[img])
            out_txt.append(self.images_text[img])
            out_mask.append(m)
        if len(out_lat) < min_contributing_images:
            return None, None, None, None

        print(f"[{tag}] Accepted: {len(out_lat)} images carry the cluster mask.", flush=True)
        return (torch.stack(out_lat), torch.stack(out_txt),
                torch.stack(out_mask), internal_cent.detach().cpu())

    @torch.no_grad()
    def _embed_tokens_external_patch(self, img_indices, device, chunk=16):
        """Decode each image, take the encoder's patch tokens and interpolate the
        patch grid onto the [grid_h, grid_w] latent-token grid. Returns
        {img_idx: [grid_h*grid_w, D]} (L2-normalized per token, row-major, CPU)."""
        if self._ext_encoder_patch is None:
            raise RuntimeError("patch granularity needs encoder_patch_fn attached.")
        mean = self._ext_mean.to(device)
        std = self._ext_std.to(device)
        ps = self._ext_patch_size
        Hp = self._ext_res // ps
        res = Hp * ps            # force divisibility so patch count == Hp*Hp exactly
        npatch = Hp * Hp
        T = self.grid_h * self.grid_w
        vdtype = next(self._ext_vae.parameters()).dtype
        out = {}
        for s in range(0, len(img_indices), chunk):
            idxs = img_indices[s:s + chunk]
            lat = torch.stack([self.images_latent[i] for i in idxs]).to(device).float()
            img = self._ext_vae.decode(lat / self._ext_vae_scale).sample
            img = (img.clamp(-1, 1) + 1.0) / 2.0
            if img.size(1) == 1:
                img = img.repeat(1, 3, 1, 1)
            img = F.interpolate(img, size=(res, res), mode="bilinear", align_corners=False)
            img = (img - mean) / std
            toks = self._ext_encoder_patch(img.to(vdtype))            # [b, S, D]
            patches = toks[:, -npatch:, :].float()                   # last npatch = patches
            D = patches.size(-1)
            grid = patches.reshape(len(idxs), Hp, Hp, D).permute(0, 3, 1, 2)  # [b,D,Hp,Hp]
            grid = F.interpolate(grid, size=(self.grid_h, self.grid_w),
                                 mode="bilinear", align_corners=False)         # [b,D,gh,gw]
            grid = grid.permute(0, 2, 3, 1).reshape(len(idxs), T, D)           # row-major
            grid = F.normalize(grid, p=2, dim=-1).cpu()
            for i, idx in enumerate(idxs):
                out[idx] = grid[i]
        return out

    def _get_clustered_dataset_external_patch(self, min_tokens_per_cluster,
                                              min_contributing_images, min_tokens_per_image,
                                              max_clusters, device, redundancy_cos_threshold,
                                              min_silhouette, max_cluster_sample=20000,
                                              silhouette_sample=4000, pca_components=64):
        """Token-wise clustering in the external space: same pipeline as the
        internal path (mean-center -> L2 -> deflate -> PCA -> spherical k-means ->
        per-token masks), but the per-token features are DINO patch features, not
        DiT activations. An expert can own a subset of an image's tokens."""
        Ntok = len(self.tok_feat)
        tag = f"Clustering/{self._ext_name}-patch"
        print(f"\n[{tag}] {Ntok} tokens buffered...", flush=True)
        if Ntok < max(min_tokens_per_cluster, max_clusters):
            print(f"[{tag}] FAILED: only {Ntok} tokens.", flush=True)
            return None, None, None, None

        # Cap distinct images so we don't decode+encode all around 19k every attempt.
        # Keep only the tokens belonging to the sampled images.
        all_img_ids = sorted(set(self.tok_img))
        M_all = len(all_img_ids)
        if M_all > self.MAX_CLUSTER_IMAGES:
            keep = set(all_img_ids[i] for i in
                       torch.randperm(M_all)[: self.MAX_CLUSTER_IMAGES].tolist())
            tok_keep = [j for j, im in enumerate(self.tok_img) if im in keep]
            tok_img_l = [self.tok_img[j] for j in tok_keep]
            tok_pos_l = [self.tok_pos[j] for j in tok_keep]
            tok_feat_l = [self.tok_feat[j] for j in tok_keep]
            img_ids = sorted(keep)
            print(f"[{tag}] {M_all} distinct images -> sampled {len(img_ids)} "
                  f"({len(tok_keep)} tokens).", flush=True)
        else:
            tok_img_l, tok_pos_l, tok_feat_l = self.tok_img, self.tok_pos, self.tok_feat
            img_ids = all_img_ids
        Ntok = len(tok_feat_l)
        tok_img = torch.tensor(tok_img_l)
        tok_pos = torch.tensor(tok_pos_l)
        id_to_row = {img: r for r, img in enumerate(img_ids)}

        # per-token DINO patch feature, aligned to buffer order. Keep the full
        # [M, T, D] stack on CPU and move only the gathered [Ntok, D] slice to the GPU.
        patch_feats = self._embed_tokens_external_patch(img_ids, device)     # img -> [T, D] CPU
        big = torch.stack([patch_feats[i] for i in img_ids])                 # [M, T, D] CPU
        rows = torch.tensor([id_to_row[i] for i in tok_img_l])              # [Ntok] CPU
        feat_dino = big[rows, tok_pos].to(device).float()                   # [Ntok, D] -> GPU
        del big

        # internal activations kept for the router-space centroid
        feat_raw_int = torch.stack(tok_feat_l).to(device).float()

        # same normalization pipeline as the internal path, on DINO features
        feat_c = feat_dino - feat_dino.mean(dim=0, keepdim=True)
        feat_norm = F.normalize(feat_c, p=2, dim=1, eps=1e-8)

        if len(self.expert_centroids_dino) > 0:
            existing = F.normalize(torch.stack(self.expert_centroids_dino).to(device).float(),
                                   p=2, dim=1, eps=1e-8)
            _, S, Vh = torch.linalg.svd(existing, full_matrices=False)
            basis = Vh[S > S[0] * 1e-3]
            feat_norm = F.normalize(feat_norm - (feat_norm @ basis.t()) @ basis, p=2, dim=1, eps=1e-8)

        pca_dim = min(pca_components, feat_norm.size(1))
        _, _, V = torch.pca_lowrank(feat_norm, q=pca_dim, center=True)
        feat_pca = F.normalize(feat_norm @ V[:, :pca_dim], p=2, dim=1, eps=1e-8)

        sub = (torch.randperm(Ntok, device=device)[:max_cluster_sample]
               if Ntok > max_cluster_sample else torch.arange(Ntok, device=device))
        fsub = feat_pca[sub]
        sil_n = min(fsub.size(0), silhouette_sample)
        fsil = fsub[torch.randperm(fsub.size(0), device=device)[:sil_n]]

        best_k, best_score, best_c = 2, -1.0, None
        for k in range(2, max_clusters + 1):
            _, cents_k = self._spherical_kmeans(fsub, k)
            sil_labels = torch.argmin(torch.cdist(fsil, cents_k), dim=1)
            score = self._pytorch_silhouette_score(fsil, sil_labels, k)
            print(f" -> k={k} | Silhouette {score:.4f}", flush=True)
            if score > best_score:
                best_score, best_k, best_c = score, k, cents_k
        labels = torch.argmin(torch.cdist(feat_pca, best_c), dim=1)
        print(f"[{tag}] optimal k={best_k} (sil={best_score:.4f}).", flush=True)
        if best_score < min_silhouette:
            print(f"[{tag}] FAILED: silhouette {best_score:.4f} < {min_silhouette}.", flush=True)
            return None, None, None, None

        existing = (F.normalize(torch.stack(self.expert_centroids_dino).to(device).float(),
                                p=2, dim=1, eps=1e-8)
                    if len(self.expert_centroids_dino) > 0 else None)

        candidates = []
        for c in range(best_k):
            cmask = (labels == c)
            n_tok = int(cmask.sum().item())
            if n_tok < min_tokens_per_cluster:
                continue
            n_img = int(torch.unique(tok_img[cmask.cpu()]).numel())
            if n_img < min_contributing_images:
                print(f"   cluster {c}: {n_tok} tok but {n_img} img -> skip.", flush=True)
                continue
            cent = F.normalize(feat_norm[cmask].mean(0), p=2, dim=0, eps=1e-8)
            novelty = (existing @ cent).max().item() if existing is not None else -1.0
            if existing is not None and novelty > redundancy_cos_threshold:
                print(f"   cluster {c}: {n_tok} tok / {n_img} img, max-cos={novelty:.3f} "
                      f"-> SKIP (covered).", flush=True)
                continue
            candidates.append((novelty, n_tok, n_img, c, cent))
            print(f"   cluster {c}: {n_tok} tok / {n_img} img, max-cos={novelty:.3f} "
                  f"-> candidate.", flush=True)

        if not candidates:
            print(f"[{tag}] FAILED: no uncovered token-cluster passes the gates.", flush=True)
            return None, None, None, None

        candidates.sort(key=lambda t: t[1], reverse=True)
        novelty, n_tok, n_img, cid, cent_dino = candidates[0]
        print(f"[{tag}] selected cluster {cid}: {n_tok} tokens / {n_img} images.", flush=True)
        self.expert_centroids_dino.append(cent_dino.detach().cpu())

        sel = (labels == cid)
        internal_cent = F.normalize(
            (feat_raw_int[sel] - feat_raw_int.mean(0, keepdim=True)).mean(0), p=2, dim=0, eps=1e-8)

        # per-token masks (subset of positions per image), exactly like internal
        sel_img = tok_img[sel.cpu()]
        sel_pos = tok_pos[sel.cpu()]
        T = self.grid_h * self.grid_w
        out_lat, out_txt, out_mask = [], [], []
        for img_idx in torch.unique(sel_img).tolist():
            positions = sel_pos[sel_img == img_idx]
            if positions.numel() < min_tokens_per_image:
                continue
            m = torch.zeros(T, dtype=torch.bool)
            m[positions] = True
            out_lat.append(self.images_latent[img_idx])
            out_txt.append(self.images_text[img_idx])
            out_mask.append(m)
        if len(out_lat) < min_contributing_images:
            return None, None, None, None

        print(f"[{tag}] Accepted: {len(out_lat)} images carry the cluster mask.", flush=True)
        return (torch.stack(out_lat), torch.stack(out_txt),
                torch.stack(out_mask), internal_cent.detach().cpu())

    def get_clustered_dataset(self, min_tokens_per_cluster=2000,
                              min_contributing_images=30, min_tokens_per_image=1,
                              max_clusters=6, device='cuda',
                              redundancy_cos_threshold=0.9, min_silhouette=0.15,
                              max_cluster_sample=20000, silhouette_sample=4000,
                              pca_components=64):
        """
        Cluster individual unsure tokens, then spawn on the largest cluster that is:
        - big enough
        - drawn from enough distinct images
        - coherent (silhouette) 
        - not already covered by an existing expert. 
        Coverage is enforced by deflation (existing directions removed before clustering) plus a
        per-candidate redundancy check.
        """
        # Opt-in: cluster in a frozen external encoder's space (DINO/CLIP) on the
        # decoded real images instead of the DiT's internal token activations.
        if getattr(self, "cluster_feature", "internal") != "internal":
            if self._ext_encoder is None:
                raise RuntimeError(
                    "cluster_feature is external but no encoder attached; call "
                    "attach_external_encoder(...) (train_phase1 does this when "
                    "--cluster_feature != internal).")
            if getattr(self, "cluster_granularity", "sample") == "patch":
                return self._get_clustered_dataset_external_patch(
                    min_tokens_per_cluster=min_tokens_per_cluster,
                    min_contributing_images=min_contributing_images,
                    min_tokens_per_image=min_tokens_per_image,
                    max_clusters=max_clusters, device=device,
                    redundancy_cos_threshold=redundancy_cos_threshold,
                    min_silhouette=min_silhouette, max_cluster_sample=max_cluster_sample,
                    silhouette_sample=silhouette_sample, pca_components=pca_components)
            return self._get_clustered_dataset_external(
                min_tokens_per_cluster=min_tokens_per_cluster,
                min_contributing_images=min_contributing_images,
                min_tokens_per_image=min_tokens_per_image,
                max_clusters=max_clusters, device=device,
                redundancy_cos_threshold=redundancy_cos_threshold,
                min_silhouette=min_silhouette)

        Ntok = len(self.tok_feat)
        print(f"\n[Clustering] Analyzing {Ntok} individual unsure TOKENS...", flush=True)
        if Ntok < max(min_tokens_per_cluster, max_clusters):
            print(f"[Clustering] FAILED: only {Ntok} tokens buffered.", flush=True)
            return None, None, None, None

        feat_raw = torch.stack(self.tok_feat).to(device).float()
        tok_img = torch.tensor(self.tok_img)
        tok_pos = torch.tensor(self.tok_pos)
        
        CLUSTER_BUFFER_CAP = 1_000_000
        if feat_raw.size(0) > CLUSTER_BUFFER_CAP:
            perm = torch.randperm(feat_raw.size(0), device=feat_raw.device)[:CLUSTER_BUFFER_CAP]
            perm_cpu = perm.cpu()
            feat_raw = feat_raw[perm]
            tok_img = tok_img[perm_cpu]
            tok_pos = tok_pos[perm_cpu]
            print(f"[Clustering] Buffer {Ntok} > cap {CLUSTER_BUFFER_CAP}; "
                  f"subsampled to {CLUSTER_BUFFER_CAP} for clustering.", flush=True)

        print("[Clustering] Applying Spherical-PCA pipeline to remove timestep variance...", flush=True)

        # Mean center (removes AdaLN shift_t)
        feat_c = feat_raw - feat_raw.mean(dim=0, keepdim=True)

        # L2 normalize (removes AdaLN scale_t)
        feat_norm = F.normalize(feat_c, p=2, dim=1, eps=1e-8)

        # Deflation: project out directions already claimed by existing
        # experts, so the same dominant axis is not rediscovered on every spawn
        # attempt once a couple of experts already cover it.
        if len(self.expert_centroids) > 0:
            existing = torch.stack(self.expert_centroids).to(device).float()   # [E, hidden]
            existing = F.normalize(existing, p=2, dim=1, eps=1e-8)

            # Orthonormalise the span first. `f - sum_j (f.e_j) e_j` is a genuine
            # projection only if the {e_j} are mutually orthonormal. With two
            # parallel directions it becomes f - 2(f.e)e, a reflection through the
            # hyperplane (magnitude preserved, sign flipped); with E near-parallel
            # directions the shared component is multiplied by (1 - E).
            #
            # SVD not QR because the centroid set may be rank-deficient (two
            # experts can legitimately cover the same direction); truncating on
            # the singular values yields an orthonormal basis for whatever
            # subspace is genuinely covered and `rank` is a useful readout.

            existing64 = existing.double()
            _, S, Vh = torch.linalg.svd(existing64, full_matrices=False)
            keep = S > (S[0] * 1e-3)
            basis = Vh[keep]                                    # [r, hidden], f64

            # Projection is done in row-chunks. The projection is row-independent, so chunking is exact.
            def _deflate_chunked(feat_f, basis_f, chunk=200_000):
                out = torch.empty_like(feat_f)
                for i in range(0, feat_f.size(0), chunk):
                    blk = feat_f[i:i + chunk]
                    out[i:i + chunk] = blk - (blk @ basis_f.t()) @ basis_f
                return out

            def _max_resid_chunked(feat_f, basis_f, chunk=200_000):
                m = 0.0
                for i in range(0, feat_f.size(0), chunk):
                    blk = feat_f[i:i + chunk]
                    m = max(m, (blk @ basis_f.t()).abs().max().item())
                return m

            fn64 = feat_norm.double()
            feat_deflated = _deflate_chunked(fn64, basis)

            # Invariant on the projection itself.
            resid = _max_resid_chunked(feat_deflated, basis)
            if resid > 1e-6:
                feat_deflated = _deflate_chunked(feat_deflated, basis)
                resid = _max_resid_chunked(feat_deflated, basis)

            feat_norm = F.normalize(feat_deflated.float(), p=2, dim=1, eps=1e-8)

            # Only a large residual means the projection is actually broken
            flag = "" if resid < 1e-3 else f"  [WARN] residual {resid:.2e} -- projection BROKEN, novelty scores unreliable!"
            print(f"[Clustering] Deflated {existing.size(0)} stored direction(s) "
                  f"spanning rank {basis.size(0)} (orthonormalised, "
                  f"max residual {resid:.2e}).{flag}", flush=True)
            if basis.size(0) < existing.size(0):
                print(f"[Clustering]   note: {existing.size(0)} centroids collapse to rank "
                      f"{basis.size(0)} -- experts are covering overlapping directions.",
                      flush=True)

        # PCA projection (removes high-dimensional soap bubble effect)
        actual_pca_dim = min(pca_components, feat_norm.size(1))
        _, _, V = torch.pca_lowrank(feat_norm, q=actual_pca_dim, center=True)
        feat_pca = torch.matmul(feat_norm, V[:, :actual_pca_dim])

        # Final spherical normalization for k-means
        feat_pca_norm = F.normalize(feat_pca, p=2, dim=1, eps=1e-8)

        if Ntok > max_cluster_sample:
            sub = torch.randperm(Ntok, device=device)[:max_cluster_sample]
        else:
            sub = torch.arange(Ntok, device=device)

        # Produces `labels` over all Ntok tokens (label -1 == noise/unclustered,
        # only ever emitted by hdbscan) and `n_groups` = number of real clusters.
        labels, n_groups, ok = self._label_tokens(
            feat_pca_norm, sub, device, max_clusters, min_silhouette,
            silhouette_sample)
        if not ok:
            return None, None, None, None
        best_k = n_groups   # kept name for the downstream loop bound

        existing = None
        if len(self.expert_centroids) > 0:
            existing = torch.stack(self.expert_centroids).to(device)
            existing = F.normalize(existing, p=2, dim=1, eps=1e-8)

        #  A cluster is a valid spawn target iff it:
        # - has enough tokens
        # - draws on enough distinct images
        # - is not already covered by an existing expert. 
        # Among valid candidates we take the largest rarit emerges as deflation peels 
        # off each covered direction over successive spawns.
        candidates = []
        for c in range(best_k):
            cmask = (labels == c)
            n_tok = int(cmask.sum().item())
            if n_tok < min_tokens_per_cluster:
                continue

            imgs_in_c = tok_img[cmask.cpu()]
            n_img = int(torch.unique(imgs_in_c).numel())
            if n_img < min_contributing_images:
                print(f"   cluster {c}: {n_tok} tokens but only {n_img} images "
                      f"(< {min_contributing_images}) -> skip (image-concentrated).", flush=True)
                continue

            # centroid in the original (deflated, normalized) space so future
            # deflation composes correctly across spawns
            cent_orig = feat_norm[cmask].mean(dim=0)
            cent_orig = F.normalize(cent_orig, p=2, dim=0, eps=1e-8)

            novelty = (existing @ cent_orig).max().item() if existing is not None else -1.0

            # per-candidate coverage filter: skip clusters an existing expert
            # already covers, rather than only rejecting the single selected one.
            if existing is not None and novelty > redundancy_cos_threshold:
                print(f"   cluster {c}: {n_tok} tokens ({n_tok/Ntok:.1%}), {n_img} images, "
                      f"max-cos={novelty:.3f} -> SKIP (already covered by an expert).", flush=True)
                continue

            frac = n_tok / Ntok
            candidates.append((novelty, n_tok, n_img, c, cent_orig))
            print(f"   cluster {c}: {n_tok} tokens ({frac:.1%} of buffer), {n_img} images, "
                  f"max-cos-to-existing={novelty:.3f} -> candidate.", flush=True)

        if not candidates:
            print(f"[Clustering] FAILED: no cluster passes the size/image gates "
                  f"AND is uncovered by existing experts.", flush=True)
            return None, None, None, None

        # Largest uncovered cluster wins (most training signal). Coverage is
        # already guaranteed by the per-candidate filter above.
        candidates.sort(key=lambda t: t[1], reverse=True)
        novelty, n_tok, n_img, cid, cent_orig = candidates[0]
        print(f"[Clustering] Selected largest uncovered cluster {cid}: "
              f"{n_tok} tokens ({n_tok/Ntok:.1%}) / {n_img} images / max-cos={novelty:.3f}.", flush=True)

        cmask = (labels == cid).cpu()
        sel_img = tok_img[cmask]
        sel_pos = tok_pos[cmask]
        T = self.grid_h * self.grid_w

        out_latents, out_texts, out_masks = [], [], []
        for img_idx in torch.unique(sel_img).tolist():
            positions = sel_pos[sel_img == img_idx]
            if positions.numel() < min_tokens_per_image:
                continue
            m = torch.zeros(T, dtype=torch.bool)
            m[positions] = True

            assert m.sum() == positions.numel(), "duplicate positions in cluster mask"
            out_latents.append(self.images_latent[img_idx])
            out_texts.append(self.images_text[img_idx])
            out_masks.append(m)

        if len(out_latents) < min_contributing_images:
            return None, None, None, None

        latents = torch.stack(out_latents)
        texts = torch.stack(out_texts)
        masks = torch.stack(out_masks)

        print(f"[Clustering] Accepted: {latents.size(0)} images carry the cluster-specific mask.", flush=True)
        return latents, texts, masks, cent_orig.detach().cpu()


    def clear(self):
        self.images_latent, self.images_text = [], []
        self._img_key_to_idx = {}
        self.tok_feat, self.tok_img, self.tok_pos = [], [], []


def train_new_expert_offline(model, moe_module, buffer_manager, new_expert_idx,
                             scheduler, device, epochs=5,
                             offline_batch_size=32,
                             offline_lr=3e-4,
                             min_tokens_per_cluster=2000,
                             min_contributing_images=30,
                             min_tokens_per_image=1,
                             min_silhouette=0.15,
                             redundancy_cos_threshold=0.9,
                             warm_start_router=True,
                             warm_start_strength=1.0):
    print(f"\n--- [SPAWNING EVENT] Attempting to spawn Expert {new_expert_idx}... ---", flush=True)

    # Seed missing centroids (once). The initial always-on experts (0, 1) were
    # never spawned through the buffer, so they have no stored centroid and the
    # coverage machinery (deflation + redundancy) would be blind to what they
    # already handle. Seed them once from the mean input token each has routed.
    if not getattr(buffer_manager, "_seeded_initial", False):
        seeded = 0
        for i in range(new_expert_idx):
            cvec = moe_module.get_expert_input_centroid(i, min_tokens=200)
            if cvec is not None:
                buffer_manager.expert_centroids.append(
                    F.normalize(cvec, p=2, dim=0, eps=1e-8))
                seeded += 1
            else:
                print(f"   [seed] initial expert {i} routed too few tokens to seed; "
                      f"not covered this attempt.", flush=True)
        buffer_manager._seeded_initial = True
        print(f"   [seed] seeded {seeded} initial-expert centroid(s) from routed-token means.", flush=True)

    c_latents, c_texts, c_masks, c_centroid = buffer_manager.get_clustered_dataset(
        min_tokens_per_cluster=min_tokens_per_cluster,
        min_contributing_images=min_contributing_images,
        min_tokens_per_image=min_tokens_per_image,
        device=device, redundancy_cos_threshold=redundancy_cos_threshold,
        min_silhouette=min_silhouette)

    if c_latents is None:
        print(f"--- [ABORT] Spawn of Expert {new_expert_idx} aborted. Buffer cleared. ---\n", flush=True)
        buffer_manager.clear()
        return

    moe_module.active_mask[new_expert_idx] = True
    moe_module.force_expert_idx = new_expert_idx
    # Clear any perf-filter mask so it can't leak into the offline
    # forwards (which use different batch sizes). The forced-expert branch skips
    # the intersection anyway. This just makes the invariant explicit.
    moe_module.latest_poorly_served_mask = None
    buffer_manager.expert_centroids.append(c_centroid)

    dataset = TensorDataset(c_latents, c_texts, c_masks)
    offline_loader = DataLoader(dataset, batch_size=offline_batch_size, shuffle=True)

    new_up_expert = moe_module.up_experts[new_expert_idx]
    new_down_expert = moe_module.down_experts[new_expert_idx]
    optimizer = torch.optim.AdamW(
        list(new_up_expert.parameters()) + list(new_down_expert.parameters()), lr=offline_lr)

    scaler = torch.amp.GradScaler('cuda')
    model.train()

    # Warm-start: collect router-input means over the offline burst so we can aim
    # the new router row at this cluster before it returns online.
    if warm_start_router:
        moe_module.ws_begin()

    diag_ratio_sum, diag_steps = 0.0, 0
    for epoch in range(epochs):
        for latents, text_embs, token_masks in offline_loader:
            latents = latents.to(device)
            text_embs = text_embs.to(device)
            token_masks = token_masks.to(device)          # [b, T] bool, cluster-specific
            bsz = latents.shape[0]
            moe_module.force_token_mask = token_masks     # [b, T]; forward uses x[token_mask]

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            target = scheduler.get_velocity(latents, noise, timesteps)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                model_output = model(
                    hidden_states=noisy_latents, encoder_hidden_states=text_embs,
                    timestep=timesteps, class_labels=torch.zeros((bsz,), dtype=torch.long, device=device)
                ).sample
                raw_loss = F.mse_loss(model_output, target, reduction='none')   # [b,C,64,64]
                grid_mask = token_masks.view(bsz, 1, buffer_manager.grid_h, buffer_manager.grid_w).float()
                spatial_mask = F.interpolate(grid_mask, scale_factor=2.0, mode='nearest')
                spatial_mask = spatial_mask.expand_as(raw_loss)
                expert_loss = (raw_loss * spatial_mask).sum() / (spatial_mask.sum() + 1e-8)

            scaler.scale(expert_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                diag_ratio_sum += float(moe_module.latest_update_ratio)
                diag_steps += 1

    print(f"--- Successfully trained Expert {new_expert_idx} "
          f"(mean offline update-ratio {diag_ratio_sum / max(1, diag_steps):.4f}). Returning Online. ---\n",
          flush=True)

    # Warm-start: aim the router at the cluster now, while force state (and the
    # collected means) is still valid. Must run before force_* are cleared.
    if warm_start_router:
        moe_module.warm_start_router_for_expert(
            new_expert_idx, strength=warm_start_strength)
    else:
        moe_module.ws_reset()

    moe_module.force_expert_idx = None
    moe_module.force_token_mask = None
    model.zero_grad(set_to_none=True)
    buffer_manager.clear()