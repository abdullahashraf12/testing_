# Production Readiness Checklist

Use this checklist as a go/no-go gate before production rollout.

## 1) Environment & Hardware
- [ ] NVIDIA drivers installed and `nvidia-smi` available.
- [ ] CUDA version validated against chosen model family profile.
- [ ] Offload storage path has sufficient free disk.
- [ ] RAM + swap strategy validated for target model size.

## 2) Runtime Configuration
- [ ] Selected runtime policy pack (`runtime.policy` / `--runtime-policy`) is documented.
- [ ] Strict compatibility mode decision is explicit (`--strict-compat` on/off).
- [ ] Precision mode is explicitly set (`fp16` / `bf16` / `fp32`).
- [ ] Swap policy is explicitly set (`required` / `preferred` / `disabled`).

## 3) Validation
- [ ] Unit test suite passes.
- [ ] Syntax checks pass.
- [ ] Scenario matrix dry-runs complete successfully.
- [ ] Startup report generated and reviewed for warnings/errors/failures.

## 4) Operational Readiness
- [ ] Failure taxonomy codes are documented for on-call responders.
- [ ] Startup report persistence location is configured and writable.
- [ ] Rollback policy for runtime config and model revision is defined.

## 5) Release Decision
- [ ] No unresolved strict compatibility errors.
- [ ] No blocker failures in startup report.
- [ ] Throughput/latency accepted for target workload.
- [ ] Final go/no-go approved by operator.

