# KQuant provenance for this experiment

This directory is a vendored snapshot of
[`local-inference-lab/kquant`](https://github.com/local-inference-lab/kquant)
based on commit:

```text
104dd9233f850a3955f4991bea68b07dd34deeb8
```

The working tree contains local experiment changes in:

- `kquant/exl3_encoder_backend.py`
- `kquant/sqg_quantizer.py`
- `tests/test_sqg_quantizer.py`

Those changes implement and test the SQG integration used by this GLM-5.2
experiment. The exact textual diff from the upstream base is preserved in
`LOCAL_CHANGES.patch`. The nested upstream `.git` directory is intentionally
not vendored, so this publication has one unambiguous Git history.

This provenance statement does not claim that upstream KQuant adopted these
local changes or that the upstream project endorses the experimental results.
