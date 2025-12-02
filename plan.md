## Plan for Protein Contact Model Integration

**Goal:** Implement `ProteinContactModule` and `ESMFoldContactNet` to consume ESM2 embeddings and contact maps, incorporating ESM Fold’s folding trunk and new pair/sequence heads (with CRF). Also update data flow to return 4-tuples (already done).

### Assumptions
- Precomputed contact maps live at `data/contact_maps/<chain_id>.pt` with keys `contact_map` (reduced L'×L'), `valid_mask` (L), `valid_indices` (len L'), `seq_len` (L).
- Dataloaders provide `(embeddings, targets, contact_map_full, valid_mask)` after padding (batch, Lmax, Lmax).
- CRF implementation from `ProteinResCRFLitModule` can be copied/reused for sequence head loss.
- ESM2 embeddings are already projected to 5120 (from `esm2_token_embeddings_sharded_fp16_sorted`) and fed as input to the model.

### Components to build
1) **ESMFoldContactNet (`src/models/components/esm_fold_contact_net.py`):**
   - Take ESM2 embeddings and apply initial projection/normalization as in `ESMTransformerResNet` up to pre-transformer stage (1D stream).
   - Build starter pair representation:
     - Relative position embedding: learnable embedding over (i - j) binned distances (e.g., clipped range) or sinusoidal. Need config for size.
     - Symmetric pair features: linear `W` on residue embeddings -> `p[i]`; combine as `p[i] + p[j]` and pass through small MLP to get pair dim.
     - Sum relative + symmetric to initialize pair rep (L×L×d_pair).
   - Folding trunk: use `FoldingTrunkConfig`/`FoldingTrunk` to process sequence (MSA/pair) streams; feed 1D stream as single-sequence MSA. Need to map outputs (pair/seq).
   - Heads:
     - Sequence head: project final seq rep to CRF emissions (num_classes).
     - Pair head: project final pair rep to contact logits (1 channel) and mask with `valid_mask`.

2) **ProteinContactModule (`src/models/protein_module.py`):**
   - Lightning module wiring model, losses, and metrics.
   - Sequence loss via linear-chain CRF (copied from `ProteinResCRFLitModule`), using padding mask from inputs.
   - Contact loss: BCE or focal on predicted contact logits vs contact_map, applied only where both residues valid (from mask).
   - Forward should accept batch tuple and return logits and loss dict.

### Tasks / Steps
1. Inspect `ProteinResCRFLitModule` to copy CRF implementation utilities.
2. Define relative position embedding scheme and pair MLP dims; add hyperparameters to `protein_contact.yaml` if needed.
3. Implement `ESMFoldContactNet` with:
   - Input projection (reuse from ResNet up to transformer prep).
   - Relative position embedding module and symmetric pair MLP.
   - FoldingTrunk wiring (configure to accept our embeddings as msa/seq and pair).
   - Heads for CRF emissions and contact logits.
4. Implement `ProteinContactModule` training/validation/test steps, losses, and metrics.
5. Quick local shape sanity check (optional minimal test script).

### Open Questions
- Exact binning range for relative position embedding (default to clipped [-k, k] -> embedding table).
- Choice of contact loss (BCEWithLogits) and weighting vs CRF loss.
- Whether to include pair-wise masking for padded positions beyond `valid_mask`.
