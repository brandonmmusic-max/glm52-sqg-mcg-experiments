# Test 8c-layer r3: combined kernel and the full-W4A8 speed-gate result

Continuation of the r2 route-packed kernel work. Two agents iterated in
parallel and the results converge here: the canonical kernel is the
M128xN64 compact-B-staged launch (independent parallel work) with the
n16 pair-decode grafted on top (this branch). All arms remain bit-exact
against the one-warp kernel by test; 16/16 GLM kernel tests pass.

## Gate/up projection progression (median, isolated fc1 component)

| Kernel state | M=3,072 | M=4,096 |
|---|---:|---:|
| one-warp M64xN8 (r1 diagnosis) | 55.613 ms | 73.065 ms |
| M64xN256 tile (r2) | 16.306 ms | 19.552 ms |
| r2 + pair-decode | 15.753 ms | 19.416 ms |
| r2 + pair-decode + hoisted loads | 15.590 ms | 19.043 ms |
| M128xN64 compact-B-staged + pair-decode (r3 canonical) | **11.166 ms** | **12.873 ms** |

Negative results preserved as evidence: pair-decode and load-hoisting gained
only 1-3% on the r2 M64xN256 kernel because that kernel was register-spill
and latency bound, not decode bound (`blocks=2` made it 1.5x worse:
`r3c` files); pipeline stages 3 and 4 lose to 2 on the r3 kernel because the
larger stages drop occupancy from two CTAs per SM to one (`r4_stages{3,4}`).
On the spill-free smem-fed M128xN64 base, the same pair-decode graft is worth
~9%.

## Full hybrid MoE layer (20 warmups, 200 balanced ABBA samples)

| Arm, M=4,096 | A16 control | Arm median | A16/arm | Amdahl e2e at f=0.31 |
|---|---:|---:|---:|---:|
| Hybrid (quality-qualified: act stays A16) | 39.57-39.78 ms | 27.48 ms | `1.4398-1.4451` | `1.105-1.107` |
| Full W4A8 (speed-only arm) | 39.78 ms | **21.81 ms** | **`1.8236`** [1.8217, 1.8245] | **`1.1628`** |

| Arm, M=3,072 | A16 control | Arm median | A16/arm | Amdahl e2e at f=0.31 |
|---|---:|---:|---:|---:|
| Hybrid | 32.19-32.61 ms | 22.60 ms | `1.4244-1.4280` | `1.102` |
| Full W4A8 (speed-only arm) | 32.61 ms | **18.58 ms** | **`1.7550`** [1.7541, 1.7565] | **`1.1539`** |

## Gate statement

- The preregistered long-prefill speed floor (1.73x full-MoE, 1.15x
  end-to-end at MoE fraction 0.31) is **met by the full-W4A8
  configuration at both M=3,072 and M=4,096**.
- The full-W4A8 arm is **speed-only and not quality-qualified**: Test 8b
  measured ~+19% routed-function damage from act-A8, and this arm's
  layer-output distortion versus dispatch-matched A16 is NMSE `9.0e-4`
  (cosine `0.99956`), 4.3x the hybrid's `2.1e-4`. The speed gate and the
  quality gate now point at different configurations; the act-A8 quality
  adjudication is the only remaining blocker for the configuration that
  passes the speed floor.
- The quality-qualified hybrid stands at `1.42-1.45x` full-MoE
  (`~1.10x` end-to-end at f=0.31). It passes 1.15x end-to-end only if the
  measured long-prefill MoE fraction is at least ~0.43, or if the gate/up
  projection reaches ~8.3 ms (a further ~1.55x; remaining identified
  levers: warp-specialized producer/consumer scheduling, smaller stages for
  3 CTAs/SM occupancy, A-fragment register pipelining).
- The declared f=0.31 remains an assumption; a workload-weighted measured
  fraction on brief-shaped prefill is required before any end-to-end claim
  is treated as serving acceptance.

Raw JSONs: `glm52_sqg_route_packed_fc1_l077_m{3072,4096}_r4_m128n64_pair.json`
(canonical projection), `..._r3{b,c,e}_*.json` (negative-result probes),
`..._r4_stages{3,4}.json` (stage-depth A/B),
`glm52_sqg_route_packed_hybrid_l077_m{3072,4096}_r4_m128n64_pair.json`
(hybrid), `..._r5_full_w4a8_arm.json` (hybrid plus the full-W4A8 speed-only
arm and its distortion). Source snapshot and hashes:
[`evaluation/w4a8_route_packed_kernel_r3/`](../evaluation/w4a8_route_packed_kernel_r3/README.md).
