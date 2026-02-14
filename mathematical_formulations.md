# Mathematical Formulations for Hierarchical World Model Analysis

This document contains the mathematical formulations that were developed for analyzing the instability mechanisms in hierarchical world models like Hieros.

## Error Propagation Analysis

### Hierarchical Objective Function
For layer $\ell$ with subgoals from layer $\ell+1$:
```latex
J^{(\ell)}(\theta^{(\ell)}) = \mathbb{E}_{\tau \sim \pi^{(\ell)}_{\theta^{(\ell)}}} \left[ \sum_{t=0}^{H-1} \gamma^t r^{(\ell)}(s_t, a_t, g_t^{(\ell+1)}) \right]
```

### Error Propagation Bound
Upper-level world model prediction error propagation to subgoal generation:
```latex
\|g_t^{\star} - g_t^{(\ell+1)}\| \leq L_g \sum_{k=0}^{t} \gamma^{t-k} \mathbb{E}[\epsilon_k^{(\ell+1)}]
```
where $\epsilon_t^{(\ell+1)} = \|s_{t+1}^{(\ell+1)} - \hat{s}_{t+1}^{(\ell+1)}\|$ and $L_g$ is the Lipschitz constant of the subgoal generation function.

### Non-stationarity Effects
Time-varying effective objective function due to simultaneous hierarchical learning:
```latex
\Delta J^{(\ell)}_t = J^{(\ell)}(\theta^{(\ell)}; \phi^{(\ell+1)}_t) - J^{(\ell)}(\theta^{(\ell)}; \phi^{(\ell+1)}_{t-\tau})
```

## Original Hieros Error Analysis

### High-level Transition and Lower-level Objective
```latex
\begin{align}
\hat z^{\mathrm{h}}_{t+1} &= f^{\mathrm{h}}_{\phi}(\hat z^{\mathrm{h}}_t, \hat a^{\mathrm{h}}_t), \\
\hat g_t &= d_{\psi}(\hat z^{\mathrm{h}}_t)
\end{align}
```

Lower-level policy objective:
```latex
J_{\mathrm{l}}(\theta_{\mathrm{l}};\phi,\psi)=
\mathbb{E}_{\tau\sim \hat p_{\phi}}\!\left[\sum_{t=0}^{H-1}\gamma^t r_{\mathrm{l}}(z_t,a_t,\hat g_t)\right]
```

### One-step Prediction Error
```latex
e_t^{\mathrm{h}}=\left\|z_{t+1}^{\mathrm{h}}-\hat z_{t+1}^{\mathrm{h}}\right\|
```

### Error Propagation to Lower Objective
Under Lipschitz continuity assumption:
```latex
\left|J_{\mathrm{l}}^{\star}(\theta_{\mathrm{l}})-J_{\mathrm{l}}(\theta_{\mathrm{l}};\phi,\psi)\right|
\le L\sum_{t=0}^{H-1}\gamma^t\,\mathbb{E}[e_t^{\mathrm{h}}]
```

### Non-stationarity Lag Term
```latex
\delta_t^{\mathrm{lag}}=
\left\|f_{\phi_t}^{\mathrm{h}}-f_{\phi_{t-\Delta}}^{\mathrm{h}}\right\|
+ D_{\mathrm{KL}}\!\left(\pi^{\mathrm{l}}_{\theta_t}\,\|\,\pi^{\mathrm{l}}_{\theta_{t-\Delta}}\right)
```

### Decomposed Objective Error Bound
```latex
\left|J_{\mathrm{l}}^{\star}-J_{\mathrm{l}}\right|
\le L\sum_t\gamma^t\mathbb{E}[e_t^{\mathrm{h}}]
+ B\sum_t\gamma^t\mathbb{E}[\delta_t^{\mathrm{lag}}]
```

## Stability Theory

### Convergence Condition
For stable hierarchical learning:
```latex
\limsup_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\epsilon_t^{(\ell+1)} + \delta_t^{\mathrm{lag}}] < \epsilon_{\text{crit}}
```
where $\epsilon_{\text{crit}}$ is the critical threshold for learning stability.

## Notes
- These formulations were developed to provide mathematical depth to the analysis of hierarchical world model instabilities
- The error propagation analysis shows how upper-level model errors compound through the hierarchy
- The non-stationarity terms capture the challenges of simultaneous multi-level learning
- Future work should focus on empirical validation of these theoretical bounds