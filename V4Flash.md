# End-to-End Logical Dataflow Graph of DeepSeek-V4-Flash

The data-centric, inference-time graph for the checked-in
[DeepSeek V4 Flash configuration](inference/config-Flash-0731.json) and
[Python reference implementation using PyTorch and TileLang](inference/model.py).
It covers the complete `Transformer.forward` path and the separately callable
`Transformer.forward_spec` DSpark path.
The supplied [generation entry point](inference/generate.py) calls only `Transformer.forward`;
DSpark is instantiated but is not used by that generation loop.

## Source authority, execution boundary, and invariants

The source order used for executable facts is:

1. [config-Flash-0731.json](inference/config-Flash-0731.json) for configured values;
2. [model.py](inference/model.py) and [kernel.py](inference/kernel.py) for actual tensor flow, layouts, precision, cache mutation, and collectives;
3. the [DeepSeek-V4 paper](2606.19348v1.pdf) for architectural intent and production-system context.

The graph models one model call in evaluation mode. The reference cache helpers implement
arbitrary-length prefill only for $p_0=0$ and single-token decode for $p_0>0$, so the decode
path assumes $S=1$. Persistent cache writes are part of the graph even though caches are not
returned. Training losses, backward propagation, Muon, and training-only load-balancing losses
are outside the inference boundary.

Two especially important source differences affecting the execution boundary are preserved here:

- The paper reports a Flash MTP depth of 1 and never names DSpark (Section 4.2.1, p. 24).
  The supplied JSON instead sets three stages, and the code implements them as
  `DSparkBlock` objects under the `mtp.\*` checkpoint namespace. This
  document calls them three configured DSpark stages, not three paper-verified MTP depths.
- The paper states a one-million-token architectural capability and later describes extending
  Flash training to a one-million-token sequence length. The JSON does not set
  `max_seq_len`: the code default is 4,096, while interactive
  `generate.py` overrides it to 65,536. A caller must explicitly construct a larger
  runtime to allocate a larger reference cache.
  (Paper: abstract and Figure 1, p. 1, and Section 4.2.2, p. 25;
  implementation: [model.py, lines 34-86](inference/model.py#L34-L86) and [generate.py, lines 80-86](inference/generate.py#L80-L86).)

## Model configuration and symbolic constants

PyTorch linear weights below use stored orientation $[\text{out features},\text{in features}]$.
A superscript $(p)$ denotes one of $P$ tensor/expert-parallel ranks.

### Dynamic and runtime dimensions

| Symbol                        | Configuration key or runtime source |                                                                                                       Effective value | Architectural role                                                           |
| ----------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------- |
| $B$                           | runtime                             |                                                                                                               dynamic | batch size                                                                   |
| $S$                           | runtime                             |                                                                              dynamic; prefill chunk, or $1$ in decode | tokens processed by this main-model call                                     |
| $S_m$                         | main hidden passed to DSpark        |                                                                                   $S$ in cache prefill, $1$ in decode | DSpark main-conditioning length                                              |
| $S_b^{\mathrm{prompt}}$       | tokenized prompt $b$                |                                                                                                               dynamic | per-example prompt length in the outer generation loop                       |
| $S_{\mathrm{new}}$            | `max_new_tokens`                    |                                                                                                               runtime | requested completion budget                                                  |
| $S_{\mathrm{total}}$          | derived generation width            |                                                        $\min(S_{\max},S_{\mathrm{new}}+\max_b S_b^{\mathrm{prompt}})$ | allocated token-buffer width                                                 |
| $p_0$                         | `start_pos`                         |                                                                                                               dynamic | absolute position of the first call token                                    |
| $N=BS$                        | derived                             |                                                                                                               dynamic | flattened token rows used by MoE                                             |
| $P$                           | distributed world size              |                                                                                                               runtime | tensor/expert-parallel rank count                                            |
| $B_{\max}$                    | `max_batch_size` default/override   |                                                                                       $4$ batch-file; $1$ interactive | allocated cache batch capacity                                               |
| $S_{\max}$                    | `max_seq_len` default/override      |                                                                                $4096$ batch-file; $65536$ interactive | allocated RoPE/cache horizon                                                 |
| $\tau$                        | `temperature`                       |                                                                                                CLI value, default $1$ | main and DSpark sampling temperature                                         |
| $C_\ell$                      | derived for $\rho_\ell>0$           |                                                                          $\left\lfloor(p_0+S)/\rho_\ell\right\rfloor$ | compressed entries computed by the end of a call                             |
| $C_\ell^{\mathrm{new}}$       | emitted in this call                | $\rho_\ell=0:0;\ \rho_\ell>0,p_0=0:\lfloor S/\rho_\ell\rfloor;\ \rho_\ell>0,p_0>0:\mathbf 1[(p_0+1)\bmod\rho_\ell=0]$ | newly emitted attention-compressor entries                                   |
| $N_{\mathrm{local}}$          | phase-dependent KV source           |                                                                                                   $p_0=0:S;\ p_0>0:W$ | physical local-KV length before any compressed region is appended/aliased    |
| $N_\ell^{kv}$                 | phase-dependent attention bank      |         $p_0=0:S+C_\ell^{\mathrm{new}};\ p_0>0,\rho_\ell=0:W;\ p_0>0,\rho_\ell>0:W+\lfloor S_{\max}/\rho_\ell\rfloor$ | physical bank length passed to sparse attention                              |
| $\bar K^{\mathrm{win}}$       | physical call width                 |                                                                                           $p_0=0:\min(S,W);\ p_0>0:W$ | fixed window-index axis for every query row in this call                     |
| $\bar K_\ell^{\mathrm{hist}}$ | physical call width                 |                                                                   $0$ (SWA); $\min(K_I,C_\ell)$ (CSA); $C_\ell$ (HCA) | fixed compressed-index axis for every query row in this call                 |
| $\bar K_\ell$                 | derived                             |                                                                   $\bar K^{\mathrm{win}}+\bar K_\ell^{\mathrm{hist}}$ | physical sparse-index/logit width; invalid query-specific slots contain $-1$ |
| $K_t^{\mathrm{win}}$          | derived                             |                                                                         $K_t^{\mathrm{win}}\leq\bar K^{\mathrm{win}}$ | valid uncompressed causal entries for query $t$                              |
| $K_{\ell,t}^{\mathrm{hist}}$  | derived                             |                                                           $K_{\ell,t}^{\mathrm{hist}}\leq\bar K_\ell^{\mathrm{hist}}$ | valid local, CSA, or HCA compressed-history entries for query $t$            |
| $K_{\ell,t}$                  | derived                             |                                                        $K_t^{\mathrm{win}}+K_{\ell,t}^{\mathrm{hist}}\leq\bar K_\ell$ | valid shared-KV candidates for query $t$ after masking padded slots          |
| $N_e$                         | runtime routing result              |                                                                                                               dynamic | token rows assigned to routed expert $e$ on one rank                         |

### Architecture, routing, and DSpark constants

| Symbol                   | Configuration key         |                      Flash value | Architectural role                                                   |
| ------------------------ | ------------------------- | -------------------------------: | -------------------------------------------------------------------- |
| $V$                      | `vocab_size`              |                         $129280$ | exact JSON vocabulary rows; the paper reports the rounded value 128K |
| $D$                      | `dim`                     |                           $4096$ | hidden width                                                         |
| $L$                      | `n_layers`                |                             $43$ | main Transformer blocks                                              |
| $L_H$                    | `n_hash_layers`           |                              $3$ | initial hash-routed MoE blocks                                       |
| $M$                      | `hc_mult`                 |                              $4$ | mHC residual streams                                                 |
| $D_{\mathrm{hc}}=MD$     | derived                   |                          $16384$ | flattened residual width                                             |
| $D_\mu=M(M+2)$           | derived                   |                             $24$ | dynamic mHC map values                                               |
| $I_{\mathrm{SK}}$        | `hc_sinkhorn_iters`       |                             $20$ | finite Sinkhorn normalization steps                                  |
| $\epsilon_n$             | `norm_eps` default        |                        $10^{-6}$ | RMS normalization stabilizer                                         |
| $\epsilon_{\mathrm{hc}}$ | `hc_eps` default          |                        $10^{-6}$ | mHC pre/Sinkhorn stabilizer                                          |
| $D_e$                    | `moe_inter_dim`           |                           $2048$ | each expert's SwiGLU width                                           |
| $E$                      | `n_routed_experts`        |                            $256$ | routed experts                                                       |
| $E_s$                    | `n_shared_experts`        |                              $1$ | replicated shared experts; code asserts one                          |
| $E_a$                    | `n_activated_experts`     |                              $6$ | selected routed experts per token                                    |
| $\phi_r$                 | `score_func`              | $\sqrt{\operatorname{softplus}}$ | router affinity function                                             |
| $\alpha_r$               | `route_scale`             |                            $1.5$ | normalized route-weight multiplier                                   |
| $c_{\mathrm{SwiGLU}}$    | `swiglu_limit`            |                             $10$ | gate upper cap and up-branch symmetric clamp                         |
| $L_D$                    | `n_mtp_layers`            |                              $3$ | configured DSpark refinement stages                                  |
| $K_D$                    | `dspark_block_size`       |                              $5$ | parallel draft positions                                             |
| $t_{\mathrm{noise}}$     | `dspark_noise_token_id`   |                         $128799$ | initialization token for draft positions $1,\ldots,K_D-1$            |
| $\mathcal T$             | `dspark_target_layer_ids` |                   $\{40,41,42\}$ | main-layer states concatenated for DSpark                            |
| $R_M$                    | `dspark_markov_rank`      |                            $256$ | Markov embedding/head rank                                           |

### Attention, position, and precision constants

| Symbol                                     | Configuration key                                       |                         Flash value | Architectural role                              |
| ------------------------------------------ | ------------------------------------------------------- | ----------------------------------: | ----------------------------------------------- |
| $N_h$                                      | `n_heads`                                               |                                $64$ | main query heads                                |
| $N_h^{(p)}=N_h/P$                          | derived                                                 |                              $64/P$ | query heads on rank $p$                         |
| $d_h$                                      | `head_dim`                                              |                               $512$ | query and shared-KV width                       |
| $d_r$                                      | `rope_head_dim`                                         |                                $64$ | trailing rotary dimensions                      |
| $d_n=d_h-d_r$                              | derived                                                 |                               $448$ | non-rotary dimensions                           |
| $R_q$                                      | `q_lora_rank`                                           |                              $1024$ | normalized query latent width                   |
| $G$                                        | `o_groups`                                              |                                 $8$ | grouped output projections                      |
| $G^{(p)}=G/P$                              | derived                                                 |                               $8/P$ | output groups on rank $p$                       |
| $H_g=N_h/G$                                | derived                                                 |                                 $8$ | query heads per output group                    |
| $R_o$                                      | `o_lora_rank`                                           |                              $1024$ | output latent width per group                   |
| $W$                                        | `window_size`                                           |                               $128$ | uncompressed causal window                      |
| $\rho_C$                                   | `compress_ratios` on CSA layers                         |                                 $4$ | overlapping CSA compression ratio               |
| $\rho_H$                                   | `compress_ratios` on HCA layers                         |                               $128$ | non-overlapping HCA compression ratio           |
| $\chi_\ell$                                | derived compressor branch count                         |            $2$ for CSA, $1$ for HCA | projected KV/gate branches before pooling       |
| $R_\ell^{\mathrm{src}}=\chi_\ell\rho_\ell$ | derived                                                 | $2\rho_C$ for CSA, $\rho_H$ for HCA | per-entry source vectors after overlap/grouping |
| $N_I$                                      | `index_n_heads`                                         |                                $64$ | lightning-indexer query heads                   |
| $N_I^{(p)}=N_I/P$                          | derived                                                 |                              $64/P$ | indexer heads on rank $p$                       |
| $d_I$                                      | `index_head_dim`                                        |                               $128$ | indexer query/key width                         |
| $K_I$                                      | `index_topk`                                            |                               $512$ | maximum selected CSA entries                    |
| $\theta_{\mathrm{local}}$                  | `rope_theta`                                            |                             $10000$ | pure-window and DSpark RoPE base                |
| $\theta_{\mathrm{comp}}$                   | `compress_rope_theta`                                   |                            $160000$ | CSA/HCA RoPE base                               |
| $S_{\mathrm{orig}}$                        | `original_seq_len`                                      |                             $65536$ | YaRN reference length for compressed layers     |
| $f_Y$                                      | `rope_factor`                                           |                                $16$ | YaRN interpolation factor                       |
| $\beta_f,\beta_s$                          | `beta_fast`, `beta_slow`                                |                              $32,1$ | YaRN correction range                           |
| $\mathcal D_w$                             | `dtype`                                                 |                            FP8 E4M3 | ordinary quantized linear weights               |
| $\mathcal D_e$                             | `expert_dtype`                                          |                     packed FP4 E2M1 | routed-expert weights                           |
| $\mathcal F_s$                             | `scale_fmt`                                             |                               UE8M0 | power-of-two scale format                       |
| $\mathcal D_s$                             | `scale_dtype` default                                   |                            FP8 E8M0 | effective activation/weight scale dtype         |
| $Q_A$                                      | [model.py, line 17](inference/model.py#L17)             |                               $128$ | activation and FP8-weight reduction block       |
| $Q_{KV}$                                   | [model.py, lines 376-378](inference/model.py#L376-L378) |                                $64$ | in-place main-KV FP8 simulation block           |
| $Q_4$                                      | [model.py, line 18](inference/model.py#L18)             |                                $32$ | FP4 simulation/weight block                     |
| $Q_S$                                      | [kernel.py, line 290](inference/kernel.py#L290)         |                                $64$ | sparse-attention gather tile                    |

### Static main-layer schedule

| Symbolic set                | Main layer IDs  | Count | Attention                                                                  | MoE routing                   |
| --------------------------- | --------------- | ----: | -------------------------------------------------------------------------- | ----------------------------- |
| $\mathcal L_{\mathrm{SWA}}$ | $0,1$           |   $2$ | shared-KV causal sliding window; $\rho_\ell=0$                             | hash table                    |
| $\mathcal L_{\mathrm{CSA}}$ | $2,4,\ldots,42$ |  $21$ | overlapping $\rho_C$ compression plus top-$K_I$ indexer and window         | layer $2$ hash; later learned |
| $\mathcal L_{\mathrm{HCA}}$ | $3,5,\ldots,41$ |  $20$ | non-overlapping $\rho_H$ compression plus all completed history and window | learned top-$E_a$             |

The 46-element `compress_ratios` array contains the 43 main ratios followed by
three zero ratios for effective DSpark layer IDs $L,\ldots,L+L_D-1$.

## Comprehensive Mermaid logical dataflow graph

Every rectangular node below is a tensor, tuple of tensors, index state, persistent buffer, or
returned data bundle. Operators, weights, casts, reshapes, branch predicates, cache writes, and
collectives are on edges. Amber nodes are persistent mutable state. Dashed-border nodes are
logical tensors that the fused kernel tile-materializes rather than allocating at full shape.

```mermaid
%%{init: {"theme": "base", "flowchart": {"htmlLabels": true, "curve": "basis", "nodeSpacing": 24, "rankSpacing": 36}}}%%
flowchart TD
    PROMPT["$$\mathcal P_{0}=\{\text{rank-0 raw prompt strings}\}$$"]
    TOK["$$\mathbf{T}^{\mathrm{call}}\in\mathbb{N}^{B\times S}\;[\mathrm{INT64}]$$"]
    POS["$$p_0\in\mathbb{N},\quad S=1\ \mathrm{if}\ p_0>0$$"]
    BUF["$$\mathbf{T}^{\mathrm{buffer}}\in\mathbb{Z}^{B\times S_{\mathrm{total}}}\;[\mathrm{INT64}],\quad -1=\text{unfilled sentinel}$$"]

    subgraph MAIN_ENTRY["Main model entry and embedding"]
        H0["$$\mathbf{H}_0\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]
        X0["$$\mathbf{X}_0\in\mathbb{R}^{B\times S\times M\times D}\;[\mathrm{BF16}]$$"]
        TOK -->|"$$\operatorname{EmbedShard}_{P}\!\left(W_{\mathrm{emb}}^{(p)}\in\mathbb{R}^{(V/P)\times D}\right);\ \operatorname{AllReduce}_{P}$$"| H0
        H0 -->|"$$\operatorname{unsqueeze}_{2};\ \operatorname{repeat}_{M}\ \text{(materialized copy)}$$"| X0
    end

    subgraph MAIN_STACK["Main decoder loop"]
        XL["$$\mathbf{X}_{\ell}\in\mathbb{R}^{B\times S\times M\times D}\;[\mathrm{BF16}],\quad 0\leq\ell<L$$"]

        subgraph MHC_ATTN["mHC around attention"]
            XAFP["$$\mathbf X_{\ell,\mathrm{flat}}^{a}\in\mathbb{R}^{B\times S\times D_{\mathrm{hc}}}\;[\mathrm{FP32}]$$"]
            MUA["$$\boldsymbol{\mu}^{a}_{\ell}\in\mathbb{R}^{B\times S\times D_{\mu}}\;[\mathrm{FP32}]$$"]
            MAPA["$$\left(\mathbf{A}^{a}_{\ell},\mathbf{C}^{a}_{\ell},\mathbf{B}^{a}_{\ell}\right)\in\mathbb{R}^{B\times S\times M}\times\mathbb{R}^{B\times S\times M}\times\mathbb{R}^{B\times S\times M\times M}\;[\mathrm{FP32}]$$"]
            UA["$$\mathbf{U}^{a}_{\ell}\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]
            HA["$$\widehat{\mathbf{H}}^{a}_{\ell}\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]
            XA["$$\mathbf{X}^{a}_{\ell}\in\mathbb{R}^{B\times S\times M\times D}\;[\mathrm{BF16}]$$"]

            XL -->|"$$\operatorname{flatten}_{M,D};\ \mathrm{BF16}\!\to\!\mathrm{FP32}$$"| XAFP
            XAFP -->|"$$\operatorname{Linear}\!\left(W_{\mathrm{hc},a}^{\ell}\in\mathbb{R}^{D_{\mu}\times D_{\mathrm{hc}}}\right)\times\operatorname{rsqrt}\!\left(\operatorname{mean}(\mathbf X^2)+\epsilon_n\right)$$"| MUA
            MUA -->|"$$\operatorname{hc\_split\_sinkhorn}_{I_{\mathrm{SK}}}\!:\ \sigma,\ 2\sigma,\ \text{row-softmax and alternating normalizations}$$"| MAPA
            XAFP -->|"$$\operatorname{view}[B,S,D_{\mathrm{hc}}]\to[B,S,M,D];\ \mathbf U^a_{\ell}=\sum_{j=0}^{M-1}A^a_{\ell,j}\mathbf X^a_{\ell,\mathrm{FP32},j};\ \mathrm{BF16}$$"| UA
            MAPA -->|"$$\mathbf A^a_{\ell}\ \text{is the reduction operand}$$"| UA
            UA -->|"$$\operatorname{RMSNorm}_{\gamma^{a}_{\ell},\epsilon_n};\ \text{FP32 statistics, BF16 output}$$"| HA
        end

        subgraph ATTN["Layer-dependent shared-KV attention"]
            QR0["$$\mathbf{Q}^{r0}_{\ell}\in\mathbb{R}^{B\times S\times R_q}\;[\mathrm{BF16}]$$"]
            QR["$$\mathbf{Q}^{r}_{\ell}\in\mathbb{R}^{B\times S\times R_q}\;[\mathrm{BF16}]$$"]
            Q["$$\mathbf{Q}_{\ell}^{(p)}\in\mathbb{R}^{B\times S\times N_h^{(p)}\times d_h}\;[\mathrm{BF16}]$$"]
            KVNOW["$$\mathbf{K\!V}^{\mathrm{now}}_{\ell}\in\mathbb{R}^{B\times S\times d_h}\;[\mathrm{BF16}]$$"]
            KVLOCAL["$$\mathbf{K\!V}^{\mathrm{local}}_{\ell}\in\mathbb{R}^{B\times N_{\mathrm{local}}\times d_h}\;[\mathrm{BF16}]$$"]
            RING["$$\mathcal K^{\mathrm{ring}}_{\ell}\equiv\mathcal K^{\mathrm{layer}}_{\ell}[:,0:W,:]\in\mathbb{R}^{B_{\max}\times W\times d_h}\;[\mathrm{BF16}]$$"]
            IWIN["$$\mathbf I^{\mathrm{win}}_{\ell}\in\mathbb{Z}^{B\times S\times\bar K^{\mathrm{win}}}\;[\mathrm{INT32}]$$"]

            HA -->|"$$\operatorname{FP8Linear}\!\left(W_{q,a}^{\ell}\in\mathbb{R}^{R_q\times D}\right);\ Q_A\text{-block activation quantization, FP32 accumulation}$$"| QR0
            QR0 -->|"$$\operatorname{RMSNorm}_{\gamma_q,\epsilon_n};\ \mathrm{FP32}\!\to\!\mathrm{BF16}$$"| QR
            QR -->|"$$\operatorname{ColumnParallelFP8Linear}\!\left(W_{q,b}^{\ell,(p)}\in\mathbb{R}^{(N_h^{(p)}d_h)\times R_q}\right);\ \operatorname{unflatten};\ \operatorname{RMS}^{-1}_{d_h}\text{ in BF16};\ \operatorname{RoPE}_{d_r}$$"| Q
            POS -->|"$$\operatorname{slice\_phase}(p_0:p_0+S)$$"| Q
            HA -->|"$$\operatorname{FP8Linear}\!\left(W_{kv}^{\ell}\in\mathbb{R}^{d_h\times D}\right);\ \operatorname{RMSNorm}_{d_h};\ \operatorname{RoPE}_{d_r};\ Q_{KV}\text{-block FP8 QDQ on }d_n$$"| KVNOW
            POS -->|"$$\operatorname{slice\_phase}(p_0:p_0+S)$$"| KVNOW
            KVNOW -->|"$$p_0=0:\ \text{current prefill KV source}$$"| KVLOCAL
            KVNOW -->|"$$\operatorname{update\_ring}_{W}(p_0)\ \text{with the last }W\text{ entries}$$"| RING
            RING -->|"$$p_0>0:\ \text{decode KV source after current-slot write}$$"| KVLOCAL
            POS -->|"$$\operatorname{causal\_window\_indices}(p_0,S,W);\ -1\text{ padding}$$"| IWIN

            CRAW["$$\mathbf C^{\mathrm{raw}}_{\ell}\in\mathbb{R}^{B\times S\times(\chi_{\ell}d_h)}\;[\mathrm{FP32}]$$"]
            ZRAW["$$\mathbf Z^{\mathrm{raw}}_{\ell}\in\mathbb{R}^{B\times S\times(\chi_{\ell}d_h)}\;[\mathrm{FP32}]$$"]
            CVAL["$$\mathbf C^{\mathrm{src}}_{\ell}\in\mathbb{R}^{B\times C_{\ell}^{\mathrm{new}}\times R_{\ell}^{\mathrm{src}}\times d_h}\;[\mathrm{FP32}]$$"]
            CGATE["$$\mathbf Z^{\mathrm{src}}_{\ell}\in\mathbb{R}^{B\times C_{\ell}^{\mathrm{new}}\times R_{\ell}^{\mathrm{src}}\times d_h}\;[\mathrm{FP32}]$$"]
            CNEW["$$\mathbf C^{\mathrm{pool}}_{\ell}\in\mathbb{R}^{B\times C_{\ell}^{\mathrm{new}}\times d_h}\;[\mathrm{FP32}]$$"]
            CEMIT["$$\mathbf C^{\mathrm{emit}}_{\ell}\in\mathbb{R}^{B\times C_{\ell}^{\mathrm{new}}\times d_h}\;[\mathrm{BF16}]$$"]
            CSTATE["$$\left(\mathcal S^{kv}_{\ell},\mathcal S^{z}_{\ell}\right)\in\mathbb{R}^{B_{\max}\times(\chi_{\ell}\rho_{\ell})\times(\chi_{\ell}d_h)}\times\mathbb{R}^{B_{\max}\times(\chi_{\ell}\rho_{\ell})\times(\chi_{\ell}d_h)}\;[\mathrm{FP32}]$$"]
            CCACHE["$$\mathcal K^{\mathrm{comp}}_{\ell}\equiv\mathcal K^{\mathrm{layer}}_{\ell}[:,W:,:]\in\mathbb{R}^{B_{\max}\times\lfloor S_{\max}/\rho_{\ell}\rfloor\times d_h}\;[\mathrm{BF16}]$$"]

            HA -->|"$$\ell\in\mathcal L_{\mathrm{CSA}}\cup\mathcal L_{\mathrm{HCA}}:\ \mathrm{FP32};\ \operatorname{Linear}\!\left(W_{c,kv}^{\ell}\in\mathbb{R}^{(\chi_{\ell}d_h)\times D}\right)$$"| CRAW
            HA -->|"$$\ell\in\mathcal L_{\mathrm{CSA}}\cup\mathcal L_{\mathrm{HCA}}:\ \mathrm{FP32};\ \operatorname{Linear}\!\left(W_{c,z}^{\ell}\in\mathbb{R}^{(\chi_{\ell}d_h)\times D}\right)$$"| ZRAW
            CRAW -->|"$$p_0=0:\ \operatorname{group/overlap}_{\rho_{\ell}}\ \text{for complete current-call blocks}$$"| CVAL
            ZRAW -->|"$$p_0=0:\ +\operatorname{APE}_{\rho_{\ell}};\ \operatorname{group/overlap}_{\rho_{\ell}}\ \text{for complete current-call blocks}$$"| CGATE
            CSTATE -->|"$$p_0>0:\ \text{updated (prior plus current) overlap/incomplete-value operand}$$"| CVAL
            CSTATE -->|"$$p_0>0:\ \text{updated overlap/incomplete-gate operand};\ -\infty\text{ for missing CSA predecessor}$$"| CGATE
            CRAW -->|"$$\text{prefill overlap/remainder snapshots or decode slot write}$$"| CSTATE
            ZRAW -->|"$$+\operatorname{APE}\text{ at the stored token slot};\ \text{prefill/decode score-state write}$$"| CSTATE
            POS -->|"$$p_0=0:\ \operatorname{select\_prefill\_overlap/remainder};\quad p_0>0:\ \operatorname{select\_decode\_slot}(p_0\bmod\rho_{\ell})$$"| CSTATE
            CVAL -->|"$$\operatorname{softmax}_{R_{\ell}^{\mathrm{src}}}\!\left(\mathbf Z^{\mathrm{src}}_{\ell}\right)\odot\mathbf C^{\mathrm{src}}_{\ell};\ \sum_{R_{\ell}^{\mathrm{src}}}$$"| CNEW
            CGATE -->|"$$\text{feature-wise gate operand for the weighted reduction}$$"| CNEW
            CNEW -->|"$$\mathrm{FP32}\!\to\!\mathrm{BF16};\ \operatorname{RMSNorm}_{d_h};\ \operatorname{RoPE}_{d_r};\ Q_{KV}\text{-block FP8 QDQ on }d_n$$"| CEMIT
            POS -->|"$$p_0=0:\ C_{\ell}^{\mathrm{new}}=\lfloor S/\rho_{\ell}\rfloor;\quad p_0>0:\ C_{\ell}^{\mathrm{new}}=\mathbf1[(p_0+1)\bmod\rho_{\ell}=0];\ \operatorname{block\_start\_phase}_{\rho_{\ell}}$$"| CEMIT
            CEMIT -->|"$$\operatorname{cache\_write}\text{ into the compressed alias view}$$"| CCACHE

            QI["$$\mathbf Q^{I,(p)}_{\ell}\in\mathbb{R}^{B\times S\times N_I^{(p)}\times d_I}\;[\mathrm{BF16\ after\ FP4\ QDQ}]$$"]
            IRAW["$$\left(\mathbf K^{I,\mathrm{raw}}_{\ell},\mathbf Z^{I,\mathrm{raw}}_{\ell}\right)\in\mathbb{R}^{B\times S\times(2d_I)}\times\mathbb{R}^{B\times S\times(2d_I)}\;[\mathrm{FP32}]$$"]
            ISTATE["$$\left(\mathcal S^{I,kv}_{\ell},\mathcal S^{I,z}_{\ell}\right)\in\mathbb{R}^{B_{\max}\times(2\rho_C)\times(2d_I)}\times\mathbb{R}^{B_{\max}\times(2\rho_C)\times(2d_I)}\;[\mathrm{FP32}]$$"]
            KI["$$\mathcal K^{I}_{\ell}\in\mathbb{R}^{B_{\max}\times\lfloor S_{\max}/\rho_C\rfloor\times d_I}\;[\mathrm{BF16\ after\ FP4\ QDQ}]$$"]
            WI["$$\boldsymbol{\omega}^{I,(p)}_{\ell}\in\mathbb{R}^{B\times S\times N_I^{(p)}}\;[\mathrm{BF16}]$$"]
            ISCORE["$$\mathbf J_{\ell}\in\mathbb{R}^{B\times S\times C_{\ell}}\;[\mathrm{BF16}]$$"]
            IEMPTY["$$\varnothing\in\mathbb{Z}^{B\times S\times 0}\;[\mathrm{INT32}]$$"]
            IHIST["$$\mathbf I^{\mathrm{hist}}_{\ell}\in\mathbb{Z}^{B\times S\times\bar K_{\ell}^{\mathrm{hist}}}\;[\mathrm{INT32}]$$"]

            QR -->|"$$\ell\in\mathcal L_{\mathrm{CSA}}:\ \operatorname{ColumnParallelFP8Linear}\!\left(W_{Iq}^{\ell,(p)}\in\mathbb{R}^{(N_I^{(p)}d_I)\times R_q}\right);\ \operatorname{RoPE}_{d_r};\ \operatorname{Hadamard}/\sqrt{d_I};\ Q_4\text{-block FP4 QDQ}$$"| QI
            POS -->|"$$\operatorname{slice\_phase}(p_0:p_0+S)$$"| QI
            HA -->|"$$\ell\in\mathcal L_{\mathrm{CSA}}:\ \mathrm{FP32};\ \left(\operatorname{Linear}_{I,kv},\operatorname{Linear}_{I,z}\right),\ \text{each }D\!\to\!2d_I$$"| IRAW
            IRAW -->|"$$\text{prefill overlap/remainder snapshots or decode slot writes};\ \mathbf Z^{I,\mathrm{raw}}+\operatorname{APE}$$"| ISTATE
            POS -->|"$$p_0=0:\ \operatorname{select\_prefill\_overlap/remainder};\quad p_0>0:\ \operatorname{select\_indexer\_decode\_slot}(p_0\bmod\rho_C)$$"| ISTATE
            IRAW -->|"$$p_0=0:\ \operatorname{overlap}_{\rho_C};\ +\operatorname{APE}\text{ before feature softmax};\ \operatorname{pool};\ \mathrm{BF16};\ \operatorname{RMSNorm};\ \operatorname{RoPE};\ \operatorname{Hadamard};\ \mathrm{FP4\ QDQ};\ \operatorname{cache\_write}$$"| KI
            ISTATE -->|"$$p_0>0:\ \text{updated (prior plus current) incomplete/overlap operand};\ \operatorname{pool/norm/RoPE/Hadamard/FP4\ QDQ};\ \operatorname{cache\_write\ at\ completion}$$"| KI
            POS -->|"$$\operatorname{block\_start\_phase}_{\rho_C}$$"| KI
            HA -->|"$$\ell\in\mathcal L_{\mathrm{CSA}}:\ \operatorname{ColumnParallelLinear}_{\mathrm{BF16}}\!\left(W_{Iw}^{\ell,(p)}\in\mathbb{R}^{N_I^{(p)}\times D}\right)/( \sqrt{d_I N_I})$$"| WI
            QI -->|"$$\operatorname{einsum}\!\left(\mathbf Q^I,\mathcal K^I\right);\ \operatorname{ReLU};\ \sum_{N_I^{(p)}};\ \operatorname{AllReduce}_{P}$$"| ISCORE
            KI -->|"$$\text{compressed index-key operand sliced as }\mathcal K^I_{\ell}[:B,:C_{\ell},:]$$"| ISCORE
            WI -->|"$$\text{signed per-index-head weighting operand}$$"| ISCORE
            ISCORE -->|"$$\ell\in\mathcal L_{\mathrm{CSA}}:\ \operatorname{causal\_mask};\ \operatorname{TopK}_{\min(K_I,C_{\ell})};\ \operatorname{offset};\ -1\text{ invalids}$$"| IHIST
            POS -->|"$$\ell\in\mathcal L_{\mathrm{HCA}}:\ \operatorname{enumerate\_completed}_{\rho_H}(p_0,S);\ \text{all visible compressed entries}$$"| IHIST
            IEMPTY -->|"$$\ell\in\mathcal L_{\mathrm{SWA}}:\ K_{\ell}^{\mathrm{hist}}=0$$"| IHIST

            IALL["$$\mathbf I_{\ell}\in\mathbb{Z}^{B\times S\times\bar K_{\ell}}\;[\mathrm{INT32}]$$"]
            KVBANK["$$\mathcal K_{\ell}\in\mathbb{R}^{B\times N_{\ell}^{kv}\times d_h}\;[\mathrm{BF16}],\quad \mathrm{key}=\mathrm{value}$$"]
            KVG["$$\widetilde{\mathbf{K\!V}}_{\ell}\in\mathbb{R}^{B\times S\times\bar K_{\ell}\times d_h}\;[\mathrm{BF16,\ padded\ logical}]$$"]
            ZATTN["$$\mathbf Z^{a,(p)}_{\ell}\in\mathbb{R}^{B\times S\times N_h^{(p)}\times\bar K_{\ell}}\;[\mathrm{FP32,\ tiled\ logical}]$$"]
            SINK["$$\mathbf z_{\ell}^{\mathrm{sink},(p)}\in\mathbb{R}^{N_h^{(p)}}\;[\mathrm{FP32}]$$"]
            OATTN["$$\mathbf O_{\ell}^{(p)}\in\mathbb{R}^{B\times S\times N_h^{(p)}\times d_h}\;[\mathrm{BF16}]$$"]
            OREL["$$\widetilde{\mathbf O}_{\ell}^{(p)}\in\mathbb{R}^{B\times S\times N_h^{(p)}\times d_h}\;[\mathrm{BF16}]$$"]
            OG["$$\mathbf O_{\ell}^{G,(p)}\in\mathbb{R}^{B\times S\times G^{(p)}\times R_o}\;[\mathrm{BF16}]$$"]
            YA["$$\mathbf Y_{\ell}^{a}\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]

            IWIN -->|"$$\operatorname{concat}_{-1}$$"| IALL
            IHIST -->|"$$\operatorname{concat}_{-1}$$"| IALL
            KVLOCAL -->|"$$p_0=0:\ \operatorname{append}\mathbf C_{\ell}^{\mathrm{emit}};\quad p_0>0:\ \text{select persistent ring-plus-compressed storage}$$"| KVBANK
            CEMIT -->|"$$p_0=0:\ \text{post-norm, post-RoPE, post-QDQ compressed entries}$$"| KVBANK
            CCACHE -->|"$$p_0>0:\ \text{persistent compressed-history region}$$"| KVBANK
            KVBANK -->|"$$\operatorname{gather}_{Q_S\text{-entry tiles}}\!\left(\mathbf I_{\ell}\right)\ \text{inside sparse kernel}$$"| KVG
            IALL -->|"$$\text{logical gather addresses shared by all }N_h^{(p)}\text{ query heads}$$"| KVG
            Q -->|"$$\mathbf Z^{a}_{\ell,t,h,j}=\left\langle\mathbf Q_{\ell,t,h},\widetilde{\mathbf{K\!V}}_{\ell,t,j}\right\rangle/\sqrt{d_h};\ \mathrm{FP32\ accumulation}$$"| ZATTN
            KVG -->|"$$\text{shared-key operand broadcast over }N_h^{(p)}$$"| ZATTN
            ZATTN -->|"$$\operatorname{online\_softmax}_{\mathrm{FP32}};\ \exp(z^{\mathrm{sink}}_{\ell,h})\text{ only in denominator};\ \operatorname{value\_GEMM}_{\mathrm{BF16}\to\mathrm{FP32}}$$"| OATTN
            KVG -->|"$$\text{the same shared entries are value operands}$$"| OATTN
            SINK -->|"$$\text{denominator-only sink operand}$$"| OATTN
            OATTN -->|"$$\operatorname{RoPE}^{-1}_{d_r}\text{ using conjugated query-position phase}$$"| OREL
            OREL -->|"$$\operatorname{view}\ [N_h^{(p)},d_h]\to[G^{(p)},H_gd_h];\ \operatorname{einsum}_{\mathrm{BF16}}\!\left(W_{o,a}^{\ell,(p)}\in\mathbb{R}^{G^{(p)}\times R_o\times(H_gd_h)}\right)$$"| OG
            OG -->|"$$\operatorname{flatten}_{G^{(p)},R_o};\ \operatorname{RowParallelFP8Linear}\!\left(W_{o,b}^{\ell,(p)}\in\mathbb{R}^{D\times(G^{(p)}R_o)}\right);\ \operatorname{AllReduce}_{P,\mathrm{FP32}};\ \mathrm{BF16}$$"| YA
        end

        YA -->|"$$\mathbf X_{\ell,k}^{a}=C_{\ell,k}^{a}\mathbf Y_{\ell}^{a}+\sum_{j=0}^{M-1}B_{\ell,j,k}^{a}\mathbf X_{\ell,j};\ \mathrm{FP32}\!\to\!\mathrm{BF16}$$"| XA
        XL -->|"$$\text{saved BF16 residual operand (alias, not copy)}$$"| XA
        MAPA -->|"$$\left(\mathbf C_{\ell}^{a},\mathbf B_{\ell}^{a}\right)\ \text{are post-mix operands}$$"| XA

        subgraph MHC_MOE["mHC around DeepSeekMoE"]
            XFFP["$$\mathbf X_{\ell,\mathrm{flat}}^{f}\in\mathbb{R}^{B\times S\times D_{\mathrm{hc}}}\;[\mathrm{FP32}]$$"]
            MUF["$$\boldsymbol{\mu}^{f}_{\ell}\in\mathbb{R}^{B\times S\times D_{\mu}}\;[\mathrm{FP32}]$$"]
            MAPF["$$\left(\mathbf A^{f}_{\ell},\mathbf C^{f}_{\ell},\mathbf B^{f}_{\ell}\right)\in\mathbb{R}^{B\times S\times M}\times\mathbb{R}^{B\times S\times M}\times\mathbb{R}^{B\times S\times M\times M}\;[\mathrm{FP32}]$$"]
            UF["$$\mathbf U^{f}_{\ell}\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]
            HF["$$\widehat{\mathbf H}^{f}_{\ell}\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]
            AFF["$$\mathbf S_{\ell}\in\mathbb{R}^{N\times E}\;[\mathrm{FP32}]$$"]
            RID["$$\mathbf R_{\ell}\in\mathbb{Z}^{N\times E_a}\;[\mathrm{INT32\ hash}/\mathrm{INT64}\ \text{top-k}]$$"]
            RW["$$\boldsymbol{\pi}_{\ell}\in\mathbb{R}^{N\times E_a}\;[\mathrm{FP32}]$$"]
            YR["$$\mathbf Y_{\ell}^{r}\in\mathbb{R}^{N\times D}\;[\mathrm{FP32}]$$"]
            YS["$$\mathbf Y_{\ell}^{s}\in\mathbb{R}^{N\times D}\;[\mathrm{BF16}]$$"]
            YF["$$\mathbf Y_{\ell}^{f}\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]
            XNEXT["$$\mathbf X_{\ell+1}\in\mathbb{R}^{B\times S\times M\times D}\;[\mathrm{BF16}]$$"]

            XA -->|"$$\operatorname{flatten}_{M,D};\ \mathrm{BF16}\!\to\!\mathrm{FP32}$$"| XFFP
            XFFP -->|"$$\operatorname{Linear}\!\left(W_{\mathrm{hc},f}^{\ell}\in\mathbb{R}^{D_{\mu}\times D_{\mathrm{hc}}}\right)\times\operatorname{rsqrt}\!\left(\operatorname{mean}(\mathbf X^2)+\epsilon_n\right)$$"| MUF
            MUF -->|"$$\operatorname{hc\_split\_sinkhorn}_{I_{\mathrm{SK}}}\!:\ \sigma,\ 2\sigma,\ \text{row-softmax and alternating normalizations}$$"| MAPF
            XFFP -->|"$$\operatorname{view}[B,S,D_{\mathrm{hc}}]\to[B,S,M,D];\ \mathbf U^f_{\ell}=\sum_{j=0}^{M-1}A^f_{\ell,j}\mathbf X^f_{\ell,\mathrm{FP32},j};\ \mathrm{BF16}$$"| UF
            MAPF -->|"$$\mathbf A^{f}_{\ell}\ \text{is the reduction operand}$$"| UF
            UF -->|"$$\operatorname{RMSNorm}_{\gamma^{f}_{\ell},\epsilon_n};\ \text{FP32 statistics, BF16 output}$$"| HF
            HF -->|"$$\operatorname{view}[B,S,D]\to[N,D];\ \operatorname{Linear}_{\mathrm{FP32}}\!\left(W_g^{\ell}\in\mathbb{R}^{E\times D}\right);\ \mathbf S_{\ell}=\sqrt{\operatorname{softplus}(\cdot)}$$"| AFF
            TOK -->|"$$\ell<L_H:\ \operatorname{flatten};\ \operatorname{lookup}\!\left(\mathrm{tid2eid}^{\ell}\in\mathbb{Z}^{V\times E_a}\right)$$"| RID
            AFF -->|"$$\ell\geq L_H:\ \operatorname{TopK}_{E_a}\!\left(\mathbf S_{\ell}+\mathbf b_{\ell}^{\mathrm{select}}\right)$$"| RID
            AFF -->|"$$\operatorname{gather}_{\mathbf R_{\ell}}\!\left(\mathbf S_{\ell}\right);\ \operatorname{normalize}_{E_a};\ \times\alpha_r$$"| RW
            RID -->|"$$\text{selected expert-ID operand}$$"| RW
            HF -->|"$$\operatorname{view}[B,S,D]\to[N,D];\ \operatorname{dispatch\_local}_{E/P};\ \mathrm{FP8\ activation}\times\mathrm{FP4\ weights}\ W_{1,e},W_{3,e},W_{2,e};\ \operatorname{clamped\ SwiGLU};\ \operatorname{scatter\_add}_{\mathrm{FP32}};\ \operatorname{AllReduce}_{P}$$"| YR
            RID -->|"$$\text{token/expert and top-slot addresses}$$"| YR
            RW -->|"$$\boldsymbol{\pi}_{\ell}\text{ multiplies the FP32 SwiGLU activation before }W_{2,e}$$"| YR
            HF -->|"$$\operatorname{view}[B,S,D]\to[N,D];\ E_s=1:\ \operatorname{FP8Expert}\!\left(W_{1,s},W_{3,s}\in\mathbb{R}^{D_e\times D},\ W_{2,s}\in\mathbb{R}^{D\times D_e}\right);\ \operatorname{clamped\ SwiGLU}$$"| YS
            YR -->|"$$\operatorname{reshape}[N,D]\to[B,S,D];\ \operatorname{add};\ \mathrm{FP32}\!\to\!\mathrm{BF16}$$"| YF
            YS -->|"$$\text{replicated shared-expert operand}$$"| YF
            YF -->|"$$\mathbf X_{\ell+1,k}=C_{\ell,k}^{f}\mathbf Y_{\ell}^{f}+\sum_{j=0}^{M-1}B_{\ell,j,k}^{f}\mathbf X_{\ell,j}^{a};\ \mathrm{FP32}\!\to\!\mathrm{BF16}$$"| XNEXT
            XA -->|"$$\text{saved BF16 residual operand (alias, not copy)}$$"| XNEXT
            MAPF -->|"$$\left(\mathbf C_{\ell}^{f},\mathbf B_{\ell}^{f}\right)\ \text{are post-mix operands}$$"| XNEXT
        end

        XFINAL["$$\mathbf X_L\in\mathbb{R}^{B\times S\times M\times D}\;[\mathrm{BF16}]$$"]
        TAP["$$\mathbf H_{\mathcal T}\in\mathbb{R}^{B\times S\times(|\mathcal T|D)}\;[\mathrm{BF16}]$$"]

        XNEXT -->|"$$\ell+1<L:\ \text{next block iteration}$$"| XL
        XNEXT -->|"$$\ell=L-1:\ \text{exit main stack}$$"| XFINAL
        XNEXT -->|"$$\ell\in\mathcal T:\ \operatorname{mean}_{M};\ \operatorname{concat}_{D}\text{ in target-layer order}$$"| TAP
    end

    X0 -->|"$$\ell=0$$"| XL

    subgraph MAIN_HEAD["Final reduction, LM head, and main return"]
        XHFP["$$\mathbf X_{L,\mathrm{flat}}\in\mathbb{R}^{B\times S\times D_{\mathrm{hc}}}\;[\mathrm{FP32}]$$"]
        MUH["$$\boldsymbol{\mu}^{h}\in\mathbb{R}^{B\times S\times M}\;[\mathrm{FP32}]$$"]
        AH["$$\mathbf A^{h}\in\mathbb{R}^{B\times S\times M}\;[\mathrm{FP32}]$$"]
        HFIN["$$\mathbf H_{\mathrm{final}}\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]
        HN["$$\widehat{\mathbf H}_{\mathrm{final}}\in\mathbb{R}^{B\times S\times D}\;[\mathrm{BF16}]$$"]
        HLAST["$$\widehat{\mathbf H}_{\mathrm{last}}\in\mathbb{R}^{B\times D}\;[\mathrm{BF16}]$$"]
        ZLOC["$$\mathbf Z_{\mathrm{main}}^{(p)}\in\mathbb{R}^{B\times(V/P)}\;[\mathrm{FP32}]$$"]
        ZMAIN["$$\mathbf Z_{\mathrm{main}}\in\mathbb{R}^{B\times V}\;[\mathrm{FP32}]$$"]
        YMAIN["$$\mathbf y_{\mathrm{main}}\in\mathbb{N}^{B}\;[\mathrm{INT64}]$$"]
        RETMAIN["$$\left(\mathbf y_{\mathrm{main}},\mathbf Z_{\mathrm{main}},\mathbf H_{\mathcal T}\right)$$"]

        XFINAL -->|"$$\operatorname{flatten}_{M,D};\ \mathrm{BF16}\!\to\!\mathrm{FP32}$$"| XHFP
        XHFP -->|"$$\operatorname{Linear}\!\left(W_{\mathrm{hc},h}\in\mathbb{R}^{M\times D_{\mathrm{hc}}}\right)\times\operatorname{rsqrt}\!\left(\operatorname{mean}(\mathbf X^2)+\epsilon_n\right)$$"| MUH
        MUH -->|"$$\mathbf A^h=\sigma(\boldsymbol{\mu}^{h}\alpha_h+\mathbf b_h)+\epsilon_{\mathrm{hc}}$$"| AH
        XHFP -->|"$$\operatorname{view}[B,S,D_{\mathrm{hc}}]\to[B,S,M,D];\ \mathbf H_{\mathrm{final}}=\sum_{j=0}^{M-1}A_j^{h}\mathbf X_{L,\mathrm{FP32},j};\ \mathrm{BF16}$$"| HFIN
        AH -->|"$$\text{head-reduction operand; no Sinkhorn/post/residual map}$$"| HFIN
        HFIN -->|"$$\operatorname{RMSNorm}_{\gamma_{\mathrm{final}},\epsilon_n}$$"| HN
        HN -->|"$$\operatorname{select}_{S-1}\ \text{because full\_logits is false}$$"| HLAST
        HLAST -->|"$$\operatorname{Linear}_{\mathrm{FP32}}\!\left(W_{\mathrm{head}}^{(p)}\in\mathbb{R}^{(V/P)\times D}\right)$$"| ZLOC
        ZLOC -->|"$$\operatorname{AllGather}_{P};\ \operatorname{concat}_{V}$$"| ZMAIN
        ZMAIN -->|"$$\tau=0:\operatorname{argmax};\quad \tau\neq0:\operatorname{softmax}_{\mathrm{FP32}}\!\left(\mathbf Z/\max(\tau,10^{-5})\right)\ \text{then exponential-race sample per rank; no ID broadcast}$$"| YMAIN
        YMAIN -->|"$$\text{return component }1$$"| RETMAIN
        ZMAIN -->|"$$\text{return component }2$$"| RETMAIN
        TAP -->|"$$\text{return component }3$$"| RETMAIN
    end

    subgraph SPEC["Separate DSpark forward_spec API"]
        DSID["$$\mathbf T_D=[\mathbf y_{\mathrm{main}},t_{\mathrm{noise}},\ldots,t_{\mathrm{noise}}]\in\mathbb{N}^{B\times K_D}\;[\mathrm{INT64}]$$"]
        MAINX["$$\mathbf H_D^{\mathrm{main}}\in\mathbb{R}^{B\times S_m\times D}\;[\mathrm{BF16}]$$"]
        DX0["$$\mathbf X^D_0\in\mathbb{R}^{B\times K_D\times M\times D}\;[\mathrm{BF16}]$$"]
        DKCACHE["$$\left\{\mathcal K^{D,\mathrm{main}}_j\in\mathbb{R}^{B_{\max}\times W\times d_h}\;[\mathrm{BF16}]\right\}_{j=0}^{L_D-1}$$"]
        NULLRET["$$\varnothing\quad\text{(forward\_spec prefill return)}$$"]

        TAP -.->|"$$\operatorname{FP8Linear}\!\left(W_D^{\mathrm{main}}\in\mathbb{R}^{D\times(|\mathcal T|D)}\right);\ \operatorname{RMSNorm}$$"| MAINX
        YMAIN -.->|"$$\operatorname{fill}_{K_D}(t_{\mathrm{noise}});\ \mathbf T_D[:,0]=\mathbf y_{\mathrm{main}}$$"| DSID
        DSID -->|"$$\operatorname{shared\ EmbedShard}_{P};\ \operatorname{AllReduce}_{P};\ \operatorname{repeat}_{M}$$"| DX0
        MAINX -->|"$$\forall j<L_D:\ \operatorname{FP8Linear}(W_{kv}^{D,j});\ \operatorname{RMSNorm};\ \operatorname{RoPE}_{d_r,\theta_{\mathrm{local}}};\ \mathrm{FP8\ QDQ};\ \operatorname{ring\_write}$$"| DKCACHE
        POS -->|"$$\operatorname{main\ phase\ slice}(p_0:p_0+S_m)$$"| DKCACHE
        DX0 -->|"$$p_0=0:\ \text{all DSpark blocks bypass mHC, draft attention, and MoE}$$"| NULLRET
        DKCACHE -->|"$$p_0=0:\ \text{cache side effect only}$$"| NULLRET

        DXJ["$$\mathbf X^D_j\in\mathbb{R}^{B\times K_D\times M\times D}\;[\mathrm{BF16}],\quad 0\leq j<L_D$$"]
        DUA["$$\widehat{\mathbf H}^{D,a}_j\in\mathbb{R}^{B\times K_D\times D}\;[\mathrm{BF16}]$$"]
        DQ["$$\mathbf Q^{D,(p)}_j\in\mathbb{R}^{B\times K_D\times N_h^{(p)}\times d_h}\;[\mathrm{BF16}]$$"]
        DKV["$$\mathbf{K\!V}^{D,\mathrm{draft}}_j\in\mathbb{R}^{B\times K_D\times d_h}\;[\mathrm{BF16}]$$"]
        DBANK["$$\mathcal K^D_j\in\mathbb{R}^{B\times(W+K_D)\times d_h}\;[\mathrm{BF16}]$$"]
        DI["$$\mathbf I^D_j\in\mathbb{Z}^{B\times K_D\times(\min(W,p_0+1)+K_D)}\;[\mathrm{INT32}]$$"]
        DO["$$\mathbf O^{D,(p)}_j\in\mathbb{R}^{B\times K_D\times N_h^{(p)}\times d_h}\;[\mathrm{BF16}]$$"]
        DYA["$$\mathbf Y^{D,a}_j\in\mathbb{R}^{B\times K_D\times D}\;[\mathrm{BF16}]$$"]
        DXA["$$\mathbf X^{D,a}_j\in\mathbb{R}^{B\times K_D\times M\times D}\;[\mathrm{BF16}]$$"]
        DUF["$$\widehat{\mathbf H}^{D,f}_j\in\mathbb{R}^{B\times K_D\times D}\;[\mathrm{BF16}]$$"]
        DYF["$$\mathbf Y^{D,f}_j\in\mathbb{R}^{B\times K_D\times D}\;[\mathrm{BF16}]$$"]
        DXNEXT["$$\mathbf X^D_{j+1}\in\mathbb{R}^{B\times K_D\times M\times D}\;[\mathrm{BF16}]$$"]
        DXEND["$$\mathbf X^D_{L_D}\in\mathbb{R}^{B\times K_D\times M\times D}\;[\mathrm{BF16}]$$"]

        DX0 -->|"$$p_0>0:\ j=0$$"| DXJ
        DXJ -->|"$$\operatorname{mHCpre}^{a}_{j};\ \operatorname{RMSNorm}\ \text{with the same FP32 map generation as the main block}$$"| DUA
        DUA -->|"$$\operatorname{FP8Linear}_{q,a};\ \operatorname{RMSNorm}_{R_q};\ \operatorname{FP8Linear}_{q,b};\ \operatorname{head\ RMS}^{-1}_{\mathrm{BF16}};\ \operatorname{RoPE}_{d_r}\text{ at }p_0+S_m,\ldots,p_0+S_m+K_D-1$$"| DQ
        POS -->|"$$\operatorname{draft\ phase\ slice}(p_0+S_m:p_0+S_m+K_D)$$"| DQ
        DUA -->|"$$\operatorname{FP8Linear}_{kv};\ \operatorname{RMSNorm};\ \operatorname{RoPE}_{d_r};\ \mathrm{FP8\ QDQ}$$"| DKV
        POS -->|"$$\operatorname{draft\ phase\ slice}(p_0+S_m:p_0+S_m+K_D)$$"| DKV
        DKCACHE -->|"$$\operatorname{concat}_{1}\ \text{with current draft KV}$$"| DBANK
        DKV -->|"$$\operatorname{concat}_{1}\ \text{after the }W\text{-slot main ring}$$"| DBANK
        POS -->|"$$[0,\ldots,\min(W,p_0+1)-1]\ \Vert\ [W,\ldots,W+K_D-1]\ \text{for every draft query}$$"| DI
        DQ -->|"$$\operatorname{sparse\_attn}\!\left(\mathcal K^D_j,\mathbf I^D_j,\mathbf z_j^{\mathrm{sink}}\right);\ \text{all }K_D\text{ draft entries are mutually visible}$$"| DO
        DBANK -->|"$$\text{shared key/value operand}$$"| DO
        DI -->|"$$\text{gather-address operand}$$"| DO
        DO -->|"$$\operatorname{RoPE}^{-1}_{d_r};\ \operatorname{grouped}\ W_{o,a}^{D,j};\ \operatorname{RowParallelFP8Linear}\ W_{o,b}^{D,j};\ \operatorname{AllReduce}_{P}$$"| DYA
        DXJ -->|"$$\operatorname{mHCpost}^{a}_{j}\!\left(\mathbf Y^{D,a}_j,\mathbf X^D_j\right)$$"| DXA
        DYA -->|"$$\text{attention-result operand}$$"| DXA
        DXA -->|"$$\operatorname{mHCpre}^{f}_{j};\ \operatorname{RMSNorm}$$"| DUF
        DUF -->|"$$\operatorname{DeepSeekMoE}_{E,E_a,E_s,D_e}\ \text{with learned routing};\ \operatorname{AllReduce}_{P}$$"| DYF
        DXA -->|"$$\operatorname{mHCpost}^{f}_{j}\!\left(\mathbf Y^{D,f}_j,\mathbf X^{D,a}_j\right)$$"| DXNEXT
        DYF -->|"$$\text{MoE-result operand}$$"| DXNEXT
        DXNEXT -->|"$$j+1<L_D:\ \text{next DSpark stage}$$"| DXJ
        DXNEXT -->|"$$j=L_D-1:\ \text{exit refinement stack}$$"| DXEND

        DH["$$\mathbf H_D\in\mathbb{R}^{B\times K_D\times D}\;[\mathrm{BF16}]$$"]
        ZDBASE["$$\mathbf Z_D^{\mathrm{base}}\in\mathbb{R}^{B\times K_D\times V}\;[\mathrm{FP32}]$$"]
        DPREFIX["$$\mathbf d_{0:i}\in\mathbb{N}^{B\times(i+1)}\;[\mathrm{INT64}],\quad \mathbf d_0=\mathbf y_{\mathrm{main}}$$"]
        MEMB["$$\mathbf E_D^M\in\mathbb{R}^{B\times K_D\times R_M}\;[\mathrm{BF16}]$$"]
        MBIAS["$$\mathbf Z_{D,i}^M\in\mathbb{R}^{B\times V}\;[\mathrm{FP32}],\quad 0\leq i<K_D$$"]
        ZD["$$\mathbf Z_D=\mathbf Z_D^{\mathrm{base}}+\mathbf Z_D^M\in\mathbb{R}^{B\times K_D\times V}\;[\mathrm{FP32}]$$"]
        DOUT["$$\mathbf y_D=[\mathbf d_0,\ldots,\mathbf d_{K_D}]\in\mathbb{N}^{B\times(K_D+1)}\;[\mathrm{INT64}]$$"]
        CONF["$$\mathbf c_D\in\mathbb{R}^{B\times K_D}\;[\mathrm{FP32,\ raw}]$$"]
        RETSPEC["$$\left(\mathbf y_D,\mathbf Z_D,\mathbf c_D\right)$$"]

        DXEND -->|"$$\operatorname{mHChead};\ \mathrm{FP32}\!\to\!\mathrm{BF16}$$"| DH
        DH -->|"$$\operatorname{RMSNorm};\ \operatorname{shared\ ParallelHead}_{P}\!\left(W_{\mathrm{head}}^{(p)}\right)\ \text{with full logits};\ \operatorname{AllGather}_{P}$$"| ZDBASE
        YMAIN -->|"$$\mathbf d_0=\mathbf y_{\mathrm{main}}$$"| DPREFIX
        DPREFIX -->|"$$\forall i<K_D:\ \operatorname{MarkovEmbedShard}_{P}\!\left(W_{M,1}^{(p)}\in\mathbb{R}^{(V/P)\times R_M}\right);\ \operatorname{AllReduce}_{P}$$"| MEMB
        MEMB -->|"$$\operatorname{select}_{i};\ \operatorname{MarkovHeadShard}_{P}\!\left(W_{M,2}^{(p)}\in\mathbb{R}^{(V/P)\times R_M}\right);\ \operatorname{AllGather}_{P}$$"| MBIAS
        ZDBASE -->|"$$\mathbf Z_D\leftarrow\mathbf Z_D^{\mathrm{base}}\ \text{as the same mutable tensor before sequential position updates}$$"| ZD
        MBIAS -->|"$$\operatorname{add\_inplace}\ \mathbf Z_D[:,i,:]\mathrel{+}=\mathbf Z_{D,i}^M\ \text{before sampling step }i$$"| ZD
        ZD -->|"$$\forall i<K_D:\ \tau=0\Rightarrow\arg\max;\ \tau\neq0\Rightarrow\operatorname{softmax}\!\left(\mathbf Z_{D,:,i,:}/\max(\tau,10^{-5})\right)\ \text{then exponential race per rank; no ID broadcast}$$"| DOUT
        DOUT -->|"$$i+1<K_D:\ \text{next Markov-conditioning token}$$"| DPREFIX
        DH -->|"$$\operatorname{concat}_{D,R_M}\!\left(\mathbf H_D,\mathbf E_D^M\right);\ \operatorname{Linear}_{\mathrm{FP32}}\!\left(W_c\in\mathbb{R}^{1\times(D+R_M)}\right)$$"| CONF
        MEMB -->|"$$\text{confidence-conditioning operand}$$"| CONF
        DOUT -->|"$$\text{return component }1$$"| RETSPEC
        ZD -->|"$$\text{return component }2$$"| RETSPEC
        CONF -->|"$$\text{return component }3$$"| RETSPEC
    end

    PROMPT -.->|"$$\text{interactive launcher: }\operatorname{broadcast\_object\_list}_{P}\text{ from rank }0;\ \operatorname{encode\_messages}\to\operatorname{tokenizer.encode};\ \operatorname{fill}(-1);\ \text{left-align IDs}$$"| BUF
    BUF -.->|"$$\operatorname{slice}\ \mathbf T^{\mathrm{buffer}}[:,p_0:p_0+S]$$"| TOK
    YMAIN -.->|"$$\operatorname{where}(\text{prompt mask},\text{ground truth},\mathbf y_{\mathrm{main}});\ \operatorname{write};\ p_0\leftarrow p_0+S$$"| BUF

    classDef input fill:#e8f3ff,stroke:#2563eb,color:#0f172a,stroke-width:1.5px;
    classDef data fill:#f8fafc,stroke:#475569,color:#0f172a,stroke-width:1.2px;
    classDef cache fill:#fff4d6,stroke:#b7791f,color:#422006,stroke-width:1.6px;
    classDef index fill:#f3e8ff,stroke:#7e22ce,color:#2e1065,stroke-width:1.3px;
    classDef logical fill:#eef2ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.2px,stroke-dasharray:5 3;
    classDef output fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:1.7px;

    class PROMPT,TOK,POS,BUF input;
    class H0,X0,XL,XAFP,MUA,MAPA,UA,HA,XA,QR0,QR,Q,KVNOW,KVLOCAL,CRAW,ZRAW,CVAL,CGATE,CNEW,CEMIT,QI,IRAW,WI,ISCORE,SINK,OATTN,OREL,OG,YA,XFFP,MUF,MAPF,UF,HF,AFF,RW,YR,YS,YF,XNEXT,XFINAL,TAP,XHFP,MUH,AH,HFIN,HN,HLAST,ZLOC,KVBANK,MAINX,DSID,DX0,DXJ,DUA,DQ,DKV,DBANK,DO,DYA,DXA,DUF,DYF,DXNEXT,DXEND,DH,ZDBASE,DPREFIX,MEMB,MBIAS data;
    class RING,CSTATE,CCACHE,ISTATE,KI,DKCACHE cache;
    class IWIN,IEMPTY,IHIST,IALL,RID,DI index;
    class KVG,ZATTN logical;
    class ZMAIN,YMAIN,RETMAIN,NULLRET,ZD,DOUT,CONF,RETSPEC output;
```

## Detailed Explanations of Operator semantics and tensor transformations

### Main call and outer generation loop

The main embedding is vocabulary-sharded. Each rank masks IDs outside its contiguous
$V/P$ range, performs its local lookup, zeroes masked rows, and all-reduces the partial
embeddings. The embedding and LM head are distinct parameters; weights are not tied. The
four-stream expansion uses `repeat`, so
$[B,S,D]\to[B,S,1,D]\to[B,S,M,D]$ is materialized rather than a broadcast view.

After blocks in $\mathcal T$, the code computes an arithmetic mean over the $M$ residual
streams and concatenates the three $[B,S,D]$ taps into
$\mathbf H_{\mathcal T}\in\mathbb R^{B\times S\times3D}$. The final mHC head is a separate
sigmoid-weighted reduction; it does not form post or residual maps and does not run Sinkhorn.
The ordinary head selects only the last sequence position, computes FP32 local vocabulary
logits, and all-gathers them. ([model.py, lines 877-926](inference/model.py#L877-L926).)

The supplied generation loop stores prompts from column zero, so they are left-aligned and
right-padded despite its "left-padded" docstring. It first calls the model on the shortest common
prompt prefix and then calls it one token at a time; ground-truth prompt tokens override sampled
tokens for longer prompts. It consumes only return component 0 and never calls
`forward_spec`. ([generate.py, lines 19-56](inference/generate.py#L19-L56).)

The exact sampler is greedy only when $\tau=0$. For every $\tau\neq0$, including an unvalidated
negative CLI value, it samples by exponential race from
$\operatorname{softmax}(\mathbf Z/\max(\tau,10^{-5}))$. This rule is shared by the main and
DSpark/Markov paths. ([model.py, lines 939-946](inference/model.py#L939-L946).)

### mHC pre/post mapping

For either attention or MoE, let
$\mathbf R[b,s,j,d]\in\mathbb R^{B\times S\times M\times D}$ be the saved BF16 residual.
The code aliases this tensor, then separately flattens and upcasts it for map generation and the
pre-reduction:

$$
\widehat{\mathbf R}
=
\frac{\operatorname{vec}(\mathbf R)_{\mathrm{FP32}}}
{\sqrt{\operatorname{mean}(\operatorname{vec}(\mathbf R)^2)+\epsilon_n}},
\qquad
\boldsymbol\mu=\widehat{\mathbf R}W_{\mathrm{hc}}^{\mathsf T}.
$$

The fused kernel splits $\boldsymbol\mu$ into $M$, $M$, and $M^2$ values:

$$
A_j=\sigma(\mu_j\alpha_0+b_j)+\epsilon_{\mathrm{hc}},
\qquad
C_k=2\sigma(\mu_{M+k}\alpha_1+b_{M+k}),
$$

and forms $B_{j,k}$ by a stabilized row softmax plus $\epsilon_{\mathrm{hc}}$, one column
normalization, and then $I_{\mathrm{SK}}-1$ row/column normalization pairs. The executable
update is

$$
U[b,s,d]=\sum_{j=0}^{M-1}A[b,s,j]R_{\mathrm{FP32}}[b,s,j,d],
$$

$$
R'[b,s,k,d]
=C[b,s,k]F(U)[b,s,d]
+\sum_{j=0}^{M-1}B[b,s,j,k]R[b,s,j,d].
$$

The $B[j,k]$ orientation follows
`comb.unsqueeze(-1) * residual.unsqueeze(-2)` followed by
`sum(dim=2)`. The paper's conceptual update is Equation (1), while the stabilized
finite algorithm above is code-specific. (Paper: Section 2.2, pp. 7-8; implementation:
[model.py, lines 652-716](inference/model.py#L652-L716) and
[kernel.py, lines 371-438](inference/kernel.py#L371-L438).)

### Attention layouts, caches, and visibility

No $[B,N_h,S,d_h]$ transpose occurs. The query stays
$[B,S,N_h^{(p)},d_h]$. Shared KV has no head axis, and one vector is used as both key and
value for every query head. The selected index tensor is also shared across heads. The full
padded logical gather $[B,S,\bar K_\ell,d_h]$ and logit tensor
$[B,S,N_h^{(p)},\bar K_\ell]$ are not allocated: the TileLang kernel gathers $Q_S$ entries
per tile, masks $-1$ slots so only $K_{\ell,t}$ entries are valid for query $t$, and keeps
scores/online-softmax state in fragments.

The active KV source differs by phase:

- Prefill ($p_0=0$): attention reads the complete current-call KV tensor and appends newly
  completed compressed entries in a temporary tensor. Independently, the last $W$ local entries
  and all completed compressed entries are written to persistent caches. Compressed indices are
  offset by $S$.
- Decode ($p_0>0,S=1$): the current local entry is written to ring slot $p_0\bmod W$, a newly
  completed compression is written when $(p_0+1)\bmod\rho_\ell=0$, and attention reads the
  persistent layout $[\text{window ring}\mid\text{compressed history}]$. Compressed indices are
  offset by $W$.

The ring and compressed-history regions are aliasing slices $[0:W]$ and $[W:]$ of one contiguous
per-layer allocation, not separately allocated buffers. The compressor receives the latter view.

The code uses a reference-specific completion-boundary rule: a compressed block becomes visible
to the query at that block's final token. The resulting entry can contain the current token but
never a future token, so the implementation remains autoregressively causal. The paper says that
a query attends only to preceding compressed blocks and cannot access other tokens in its own
compressed block (Section 2.3.3, p. 13), while the boundary in Equation (16) depends on its token
indexing convention. The prose and equations therefore do not cleanly resolve this exact boundary,
and the graph follows the executable code. ([model.py, lines 260-282 and
442-548](inference/model.py#L260-L548); paper: Sections 2.3.1-2.3.3, pp. 9-13.)

For a local head $h$, the fused attention computes

$$
a_{t,h,j}
=
\frac{\exp\!\left(q_{t,h}^{\mathsf T}kv_j/\sqrt{d_h}\right)}
{\exp(z^{\mathrm{sink}}_h)+
\sum_{k\in\mathcal I_{\ell,t}}
\exp\!\left(q_{t,h}^{\mathsf T}kv_k/\sqrt{d_h}\right)}.
$$

The sink has no value vector, so real attention weights can sum below one. Scores, running
maxima, denominators, and output accumulators are FP32; score tiles are cast to BF16 for the
value GEMM, and the final output is BF16. If $N_h^{(p)}<16$, the wrapper pads to 16 heads,
then narrows and calls `contiguous()`. ([kernel.py, lines
276-368](inference/kernel.py#L276-L368); paper: Equation (27), p. 13.)

### CSA/HCA compression and the lightning indexer

Compression gates are feature-wise. With projected values $C_{r,f}$ and gate logits
$Z_{r,f}$ for source position $r$ and feature $f$:

$$
\alpha_{r,f}
=
\frac{\exp(Z_{r,f}+\operatorname{APE}_{r,f})}
{\sum_{u\in\mathcal B}\exp(Z_{u,f}+\operatorname{APE}_{u,f})},
\qquad
C_f^{\mathrm{new}}=\sum_{r\in\mathcal B}\alpha_{r,f}C_{r,f}.
$$

HCA uses one non-overlapping $\rho_H$-token source block. CSA projects two $d_h$ branches.
For compressed block $i$, the implementation's overlap transform combines one branch from
block $i-1$ with the other branch from block $i$, for $2\rho_C$ source vectors in total; the
missing predecessor of block zero has zero values and $-\infty$ gate logits. Output length
still shrinks by $\rho_C$. In decode, FP32 state buffers retain incomplete values and scores.

The CSA indexer has its own overlap-$\rho_C$ compressor and a distinct pair of FP32 incomplete
state buffers of shape $[B_{\max},2\rho_C,2d_I]$; it does not reuse the attention compressor's
state. Its score is

$$
J_{t,c}
=
\sum_{h=1}^{N_I}
\omega^I_{t,h}\,
\operatorname{ReLU}\!\left(
\left\langle q^I_{t,h},k^I_c\right\rangle
\right),
\qquad
\omega^I\ \text{includes}\ d_I^{-1/2}N_I^{-1/2}.
$$

Rank-local head contributions are all-reduced before causal masking and top-$K_I$ selection.
HCA has no learned indexer and enumerates every completed compressed entry. The first two
layers have no compressed history. (Paper: Equations (9)-(26) and Figures 3-4, pp. 9-12;
implementation: [model.py, lines 285-439](inference/model.py#L285-L439).)

### RoPE and YaRN formulation

Adjacent coordinate pairs are reinterpreted as complex values:

$$
(x_0,x_1,x_2,x_3,\ldots)
\mapsto
(x_0+i x_1,x_2+i x_3,\ldots).
$$

For $j=0,\ldots,d_r/2-1$, base frequency is

$$
\omega_j=\theta^{-2j/d_r}.
$$

When $S_{\mathrm{orig}}>0$, the code computes

$$
\ell_c=
\max\!\left(
\left\lfloor
\frac{d_r\log(S_{\mathrm{orig}}/(2\pi\beta_f))}
{2\log\theta}
\right\rfloor,0
\right),
$$

$$
h_c=
\min\!\left(
\left\lceil
\frac{d_r\log(S_{\mathrm{orig}}/(2\pi\beta_s))}
{2\log\theta}
\right\rceil,d_r-1
\right),
$$

$$
s_j=
1-\operatorname{clip}\!\left(\frac{j-\ell_c}{h_c-\ell_c},0,1\right),
\qquad
\widetilde\omega_j=
\frac{\omega_j}{f_Y}(1-s_j)+\omega_j s_j,
\qquad
f_{t,j}=e^{it\widetilde\omega_j}.
$$

Compressed layers use $(\theta,S_{\mathrm{orig}})=(\theta_{\mathrm{comp}},S_{\mathrm{orig}})$.
Pure-window and DSpark layers disable interpolation and use
$\theta_{\mathrm{local}}$. Main queries and local KV use positions
$p_0,\ldots,p_0+S-1$; a compressed entry uses its block-start position. Only the trailing
$d_r$ dimensions are cast to FP32, complex-multiplied, and copied back in place. Core attention
output uses the conjugate phase at the query position. The paper describes this negative-position
correction but uses an overloaded index symbol; the code unambiguously uses the query-position
frequency. For CSA and HCA, partial query/KV/output RoPE and the attention sink agree with
Section 2.3.3 of the paper. The reference additionally applies the same inverse-RoPE/sink
machinery to pure-window and DSpark attention paths; the paper does not specify those cases.
([model.py, lines 205-250 and 368-378](inference/model.py#L205-L378); paper:
Section 2.3.3, p. 13.)

### MoE routing and expert arithmetic

All $E$ FP32 affinities are computed even for hash-routed layers because the unbiased affinity
still supplies the mixture weight:

$$
s_e(x)=\sqrt{\operatorname{softplus}
\left(xW_{g,e}^{\mathsf T}\right)}.
$$

For $\ell<L_H$, IDs come from the fixed $V\times E_a$ token-to-expert table. For later
layers, an FP32 bias changes only top-$E_a$ selection. Selected weights are gathered from
unbiased $s_e$, normalized over $E_a$, and multiplied by $\alpha_r$.

For selected expert $e$:

$$
g_e=xW_{1,e}^{\mathsf T},
\qquad
u_e=xW_{3,e}^{\mathsf T},
$$

$$
v_e=
\operatorname{SiLU}\!\left(\min(g_e,c_{\mathrm{SwiGLU}})\right)
\odot
\operatorname{clip}\!\left(u_e,-c_{\mathrm{SwiGLU}},c_{\mathrm{SwiGLU}}\right),
\qquad
y_e=(\pi_e v_e)W_{2,e}^{\mathsf T}.
$$

Each rank instantiates a contiguous $E/P$ expert-ID range, selects local token rows with
`torch.where`, scatter-adds BF16 expert outputs into an FP32 dense tensor, and
all-reduces that tensor. The replicated shared FP8 expert is added afterward. This reference
path has no all-to-all dispatch. ([model.py, lines 551-649](inference/model.py#L551-L649).)

### DSpark API semantics

The DSpark path is an exposed proposal API, not an unconditional tail of
`Transformer.forward`.

- **Prefill, $p_0=0$.** Stage zero projects the three target-layer taps to
  $\mathbf H_D^{\mathrm{main}}$. Every DSpark stage projects that main hidden sequence into its
  own local KV ring. The draft tensor passes through all stages unchanged; mHC, draft attention,
  MoE, the output head, Markov chain, and confidence head are bypassed, and
  `forward_spec` returns `None`.
- **Decode, $p_0>0$.** The expected main conditioning length is $S_m=1$.
  Each of $L_D$ stages runs a full mHC-attention-mHC-MoE block. Five draft queries/entries use
  future positions $p_0+1,\ldots,p_0+K_D$ and all five draft entries are mutually visible in
  each stage; there is no triangular mask within the draft block.
- **Heads.** After the last stage, a DSpark-specific mHC head and norm feed the
  shared main vocabulary head with full logits. A sequential rank-$R_M$ Markov embedding/head
  adds a vocabulary bias conditioned on the previously sampled draft token. The confidence head
  is one FP32 linear over $[\mathbf H_D\mid\mathbf E_D^M]$ and returns raw, uncalibrated scores.

The return shapes are $[B,K_D+1]$ IDs, $[B,K_D,V]$ mutated base-plus-Markov logits, and
$[B,K_D]$ confidence scores. The repository contains no proposal acceptance, rejection, or
verification controller; an external scheduler would be required for speculative decoding.
([model.py, lines 743-874 and 928-936](inference/model.py#L743-L936).)

## Reshape, copy, and contiguity map

| Data transition     | Exact mapping                                     | Storage behavior                                                            |
| ------------------- | ------------------------------------------------- | --------------------------------------------------------------------------- |
| embedding expansion | $[B,S,D]\to[B,S,1,D]\to[B,S,M,D]$                 | `unsqueeze` view, then `repeat` copy                                        |
| mHC generator       | $[B,S,M,D]\to[B,S,MD]$                            | `flatten(2)`, then `float()` allocation                                     |
| mHC pre-reduction   | FP32 $[B,S,MD]\to[B,S,M,D]$                       | `view(shape)` of the upcast tensor                                          |
| main query          | $[B,S,N_h^{(p)}d_h]\to[B,S,N_h^{(p)},d_h]$        | `unflatten` view when strides permit                                        |
| indexer query       | $[B,S,N_I^{(p)}d_I]\to[B,S,N_I^{(p)},d_I]$        | `unflatten` view when strides permit                                        |
| CSA grouping        | $[B,C,\rho_C,2d]\to[B,C,2\rho_C,d]$               | explicit newly allocated overlap tensor                                     |
| HCA grouping        | $[B,C,\rho_H,d]$                                  | `unflatten` view before reduction                                           |
| attention grouping  | $[B,S,N_h^{(p)},d_h]\to[B,S,G^{(p)},H_gd_h]$      | `view`; no sequence/head transpose                                          |
| output latent       | $[B,S,G^{(p)},R_o]\to[B,S,G^{(p)}R_o]$            | `flatten(2)` view when possible                                             |
| MoE rows            | $[B,S,D]\leftrightarrow[BS,D]$                    | `view`                                                                      |
| target tap          | $[B,S,M,D]\to[B,S,D]$                             | reduction allocation; three taps concatenate to $[B,S,3D]$                  |
| index helpers       | runtime arguments $\to[B,S,K]$                    | returned tensors explicitly call `contiguous()`                             |
| sparse-head padding | $N_h^{(p)}\to16$ only if $N_h^{(p)}<16$           | concatenate zeros; output narrows and calls `contiguous()`                  |
| in-place QDQ        | BF16 $\to$ FP8/FP4 quantized temporary $\to$ BF16 | quantized temporary allocated, then copied back into the input slice/tensor |

## Precision, collectives, and numerical-risk audit

| Region                        | Stored/input data                       | Executable compute and reduction                                                         |
| ----------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------- |
| embedding                     | BF16 local $[V/P,D]$                    | local lookup, BF16 all-reduce                                                            |
| mHC maps/post mix             | BF16 residual; FP32 parameters          | FP32 RMS, linear, sigmoid, Sinkhorn, weighted sums; BF16 boundary                        |
| FP8 `Linear` projection paths | FP8 E4M3 weights with E8M0 scales       | $Q_A$-block FP8 activation quantization, FP32 scale-corrected accumulation, BF16 output  |
| grouped attention $W_{o,a}$   | BF16 weight and BF16 attention input    | grouped BF16 `einsum`, BF16 output                                                       |
| compressor                    | FP32 weights, APE, and incomplete state | FP32 projections, feature-wise softmax, weighted reduction; BF16 norm/cache output       |
| reference KV cache            | physically BF16                         | leading $d_n$ dimensions undergo in-place FP8 QDQ; trailing $d_r$ stay BF16              |
| reference indexer Q/K         | physically BF16 after QDQ               | Hadamard plus in-place FP4 QDQ; PyTorch score tensor remains BF16; score all-reduce      |
| sparse attention              | BF16 Q/KV/output                        | FP32 logits, online max/sum, sink denominator, and value accumulator                     |
| routed experts                | FP8 activations, packed FP4 weights     | FP32 scale-corrected GEMM accumulators; FP32 nonlinear/route weighting and dense combine |
| shared expert                 | FP8 weights                             | same clamped SwiGLU topology; BF16 expert output                                         |
| routed combine                | BF16 per-expert outputs                 | FP32 scatter-add and all-reduce; BF16 block boundary                                     |
| main/Markov heads             | FP32 sharded weights                    | FP32 logits and vocabulary all-gather                                                    |
| sampling                      | FP32 global logits                      | FP32 softmax/exponential-race or argmax                                                  |

Numerically sensitive reductions are deliberately visible in the graph. `RMSNorm` and
mHC reciprocal-RMS calculations, mHC sigmoid/Sinkhorn denominators, compressor softmax, sparse-attention
running maxima and denominator, sparse value accumulation, routed dense accumulation, routed
all-reduce, final logits, and sampling softmax use FP32 where the implementation explicitly
does so. The per-head query RMS rescaling and lightning-indexer einsum/head reduction have no
explicit FP32 upcast and therefore remain BF16-sensitive paths in this reference.

The distributed communication points are:

- embedding all-reduce;
- row-parallel attention-output FP32 all-reduce;
- CSA index-score all-reduce;
- routed-MoE FP32 all-reduce;
- main and Markov vocabulary-head all-gathers;
- Markov-embedding all-reduce; and
- the interactive launcher's one prompt-object broadcast from rank 0 per interactive turn, before
  tokenization and generation.

Sampled main and DSpark token IDs are not broadcast. Every rank samples independently after the
vocabulary-logit all-gather; the launcher seeds all ranks identically and therefore relies on
synchronized RNG consumption and execution rather than explicit sampled-token synchronization.

There is no pipeline-parallel or all-to-all expert path in these files. Required sharding
compatibility is $P\mid V$, $P\mid N_h$, $P\mid N_I$, $P\mid G$, and $P\mid E$; some
conditions are asserted indirectly by sharded linear sizes and later reshapes.

### Implemented fusion versus candidates

Objective implementation facts:

- `sparse_attn_kernel` fuses indexed tile gathering, dot products, online softmax,
  attention-sink handling, value accumulation, and BF16 output.
- `hc_split_sinkhorn_kernel` fuses map splitting, pre/post sigmoid constraints, and
  finite Sinkhorn normalization.
- Activation quantization and quantized GEMM are dedicated TileLang kernels, but they are
  separate Python-level calls in `linear`; this document does not call them one fused
  launch.

Engineering candidates, not implemented facts, are RMSNorm plus projection, RoPE plus cache
write, compressor projection/gating/pooling, grouped $W_{o,a}$ plus $W_{o,b}$, and a
dispatch/expert/combine mega-kernel. These candidates require compiler/runtime work beyond the
checked-in reference.

## Paper-to-reference reconciliation

The architecture in the graph agrees with the paper on the mHC residual topology, dynamic-map
concept, sigmoid constraints, expansion factor, and configured iteration count; the executable
finite Sinkhorn realization remains code-specific as documented in Section 4.2 above. It also
agrees on 43 Flash blocks, the two initial window layers, interleaved CSA/HCA, shared-key/value
MQA, partial/inverse RoPE and the attention sink in CSA/HCA, DeepSeekMoE, hash routing in the
first three MoE layers, and the core Flash dimensions enumerated in Section 4.2.1.
(DeepSeek-V4 paper: Figure 2 and Sections 2.1-2.3, pp. 6-13; Flash setup, p. 24.)

The paper also describes a fuller production stack that is not this reference execution:

| Paper production design                                                                                                                   | Checked-in reference behavior represented above                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| physically mixed KV storage: BF16 rotary tail and FP8 non-rotary dimensions                                                               | physical BF16 cache with in-place FP8 quantize/dequantize simulation                  |
| native FP4 lightning-indexer QK during inference/rollout                                                                                  | BF16 Q/K tensors after FP4 quantize/dequantize simulation, followed by PyTorch einsum |
| expert waves with dispatch and combine all-to-all in a mega-kernel                                                                        | all tokens on every rank, local expert loop, dense scatter-add, final all-reduce      |
| heterogeneous block-managed classical/state caches, plus an on-disk shared-prefix cache with deployment-selectable SWA storage strategies | one contiguous per-layer ring-plus-compressed tensor and in-memory incomplete state   |

Separately, the paper reports MTP depth $1$ in Section 4.2.1 (p. 24), whereas the checked-in
configuration and code implement three DSpark refinement stages under the `mtp.\*`
namespace. The paper does not establish that DSpark is its MTP implementation, so these are not
treated as equivalent concepts.

The production distinctions come from the paper's Sections 2.3.4, 3.1, 3.5, and 5.2.1
(pp. 13, 15-16, 21-23, and 33-34). They are not silently projected onto the reference graph.

## Implementation anchors

- Configuration: [config-Flash-0731.json](inference/config-Flash-0731.json)
- Model arguments, sharded embedding/linears, RMSNorm: [model.py, lines 34-202](inference/model.py#L34-L202)
- RoPE and index helpers: [model.py, lines 205-282](inference/model.py#L205-L282)
- Compressor and lightning indexer: [model.py, lines 285-439](inference/model.py#L285-L439)
- Main attention: [model.py, lines 442-548](inference/model.py#L442-L548)
- Router, experts, and MoE: [model.py, lines 551-649](inference/model.py#L551-L649)
- Block mHC: [model.py, lines 652-716](inference/model.py#L652-L716)
- Main/Markov vocabulary heads and DSpark: [model.py, lines 719-874](inference/model.py#L719-L874)
- Full model forward APIs and sampler: [model.py, lines 877-946](inference/model.py#L877-L946)
- Quantization, sparse attention, mHC Sinkhorn, and FP4 GEMM:
  [kernel.py](inference/kernel.py)
- Actual CLI generation loop: [generate.py, lines 19-56](inference/generate.py#L19-L56)
- Paper architecture and Flash setup: [2606.19348v1.pdf](2606.19348v1.pdf), Sections
  2.1-2.3 and 4.2.1, Figures 2-4, pp. 6-13 and 24
