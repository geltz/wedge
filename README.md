Merge method for two diffusion models.

WEDGE first winsorizes A and B to suppress outliers, computes a confidence gate from cosine similarity and a two-way entropy score to choose an adaptive temperature t, blends the tensors via a temperature-scaled log-sum-exp weighted by α, then re-standardizes to A’s mean/variance and applies a MAD-based trust clamp.

Uses the [sd_mecha](https://github.com/ljleb/sd-mecha) api.
