# Toy Diffuser Simple

Closed-form diffusion toy planner 실험 저장소입니다.  
동일한 reward에서 `non-guided`, `guided`, `tdp`, `driftlite`를 비교하고, SNIS tilted estimate와의 차이를 평가합니다.

## 1. Project Structure

```text
scripts/
  run_model.py           # 단일 모델 실행 + 통계 비교 저장
  run_eval.py            # 다중 모델 일괄 실행
  run_eval_matched.py    # 모델별 후보수(K) 맞춘 비교 실행
  visualize.py           # 샘플 분포/trajectory 시각화

src/
  config/spec.py         # 전체 설정 스키마(dataclass)
  models/
    gmm.py               # base isotropic GMM
    diffusion.py         # reverse diffusion sampler (guided/non-guided)
    tdp.py               # TDP sampler
    driftlitelite.py     # DriftLite sampler (SMC + control drift)
  planners/base.py       # sampler 공통 planner adapter
  pipelines/
    experiment_runtime.py# 모델 팩토리/런타임 구성
    run_eval_pipeline.py # 평가 파이프라인 + 결과 저장
  evaluation/moment_eval.py
                         # tilted SNIS vs model mean 비교
  tasks/reward.py        # toy reward 정의
```

### 실행 흐름
1. `experiment_runtime.build_runtime()`이 공통 구성요소(GMM, reward, schedule)와 모델 planner를 만든다.
2. 평가 파이프라인이 모델별 `sample()` 시간을 측정하며 샘플을 생성한다.
3. `moment_eval.run_eval()`이 tilted SNIS와 모델 샘플 통계를 비교한다.
4. 파이프라인이 `results.json`, `summary.csv`, `timing_summary.csv`, `npz`, `plots`를 저장한다.

## 2. Setup

```bash
conda env create -f environment.yml
conda activate toy_diffuser
```

## 3. Run

### 단일 모델
```bash
python scripts/run_model.py model.name=guided eval.device=cpu
python scripts/run_model.py model.name=driftlite eval.device=cpu
```

출력:
- `outputs/run_model/YYYY-MM-DD/HH-MM-SS/config.yaml`
- `outputs/run_model/YYYY-MM-DD/HH-MM-SS/results.json`
- (옵션) `outputs/run_model/.../data/driftlite_diagnostics.npz`

### 다중 모델 평가
```bash
python scripts/run_eval.py eval.device=cpu
python scripts/run_eval.py eval.device=cpu eval.models='[non-guided,guided,tdp,driftlite]'
```

출력:
- `outputs/run_eval/.../config.yaml`
- `outputs/run_eval/.../results.json`
- `outputs/run_eval/.../data/summary.csv`
- `outputs/run_eval/.../data/timing_summary.csv` (모델별 샘플링 시간 요약)
- `outputs/run_eval/.../data/derived_saved_subsets.npz`
- (옵션) `outputs/run_eval/.../data/driftlite_diagnostics.npz`
- `outputs/run_eval/.../plots/*.png`

### 시각화
```bash
python scripts/visualize.py eval.device=cpu
python scripts/visualize.py eval.device=cpu viz.models='[guided,driftlite]' viz.path_model=driftlite viz.n_traj=10
```

## 4. Models

- `non-guided`: pure reverse diffusion
- `guided`: reward gradient guidance
- `tdp`: parent-child mutate-select (always guided variant)
- `driftlite`: SMC + ESS resampling + VCG-style control drift

## 5. Key Configuration (default)

### traj.*
- `traj.horizon_T=32`: trajectory length
- `traj.action_dim=2`: action/state dimension
- `traj.base_action_mean=0.1`: base component mean magnitude
- `traj.sigma0_sq=0.025`: base isotropic variance
- `traj.pi_pos=0.5`: mixture ratio for positive mode

### reward.*
- `reward.w_neg=0.2`
- `reward.w_pos=0.7`
- `reward.offset=0.1`
- `reward.state_var=0.25`
- `reward.goal=None` (if set, overrides `goal_x/goal_y`)
- `reward.goal_x=None`, `reward.goal_y=None`

### diffusion.*
- `diffusion.n_steps=256`
- `diffusion.beta_start=1e-4`
- `diffusion.beta_end=2e-2`

### guidance.*
- `guidance.enabled=true`
- `guidance.scale=10.0`
- `guidance.clip_norm=1.0`

### reverse.*
- `reverse.n_candidates=1`: `non-guided/guided`에서 rollout당 후보 샘플 개수

### eval.*
- `eval.seed=0`
- `eval.device=auto`
- `eval.n_base=50000`
- `eval.n_guided=20000`
- `eval.bootstrap_reps=200`
- `eval.batch_size=131072`
- `eval.f_list=['final_x','final_y','pos_indicator','R']`
- `eval.models=[]` (empty면 등록된 전체 모델 실행)

### tdp.*
- `tdp.n_roots=64`
- `tdp.renoise_frac=0.15`

### driftlite.*
- `driftlite.n_particles=512`: particle count
- `driftlite.rollout_batch=0`: 동시 rollout 개수 (`<=0`이면 `eval.batch_size // n_particles`로 자동 계산)
- `driftlite.ess_threshold_ratio=0.5`: ESS resample threshold ratio
- `driftlite.resample=true`: ESS 기반 systematic resampling 사용
- `driftlite.dt_mode='auto'`: `auto|fixed`
- `driftlite.dt=0.0`: `dt_mode=fixed`일 때만 사용
- `driftlite.ctrl_scale=1.0`: control drift 강도
- `driftlite.basis=['grad_r','score']`: control basis (`x` 추가 가능)
- `driftlite.proxy_gamma=1.0`: proxy score 계수
- `driftlite.divergence_mode='grad_r_hutch'`: `none|grad_r_hutch|all_hutch`
- `driftlite.hutch_samples=1`: Hutchinson trace 샘플 수
- `driftlite.reward_path='constant'`: `constant|linear`
- `driftlite.reward_scale=1.0`: reward scaling
- `driftlite.reg_lambda=1e-4`: VCG linear solve regularizer
- `driftlite.output_mode='best'`: `best|resampled|weighted`
  - `best`: 각 rollout에서 `n_particles` 경쟁 후 top-1 채택, 이를 `eval.n_guided`개 반복
- `driftlite.save_diagnostics=false`: drift diagnostics 수집 on/off
- `driftlite.seed_offset=3000`: `eval.seed + seed_offset`로 drift sampler seed 생성

### output.*
- `output.root='outputs'`
- `output.timezone='Asia/Seoul'`
- `output.save_base_max=50000`
- `output.save_chain_max=20000`
- `output.save_plots_max_points=50000`
- `output.save_diagnostics=false`: diagnostics 파일 저장 on/off

실행 표시:
- `run_model.py`, `run_eval.py`, `visualize.py`는 공통 배치 루프에서 `tqdm` 진행바만 표시합니다.
- `run_eval.py`는 결과 표 아래에 모델별 `Sampling Time by Model` 표를 함께 출력합니다.

### visualize.*
- `viz.n_samples=200` (기본): 시각화용 샘플 수. `eval.n_guided`와 분리되어 있어 시각화가 과도하게 느려지지 않도록 함.

## 6. DriftLite Diagnostics

`driftlite` 실행 시 diagnostics 수집은 기본 off입니다.

수집/저장 활성화:
- `driftlite.save_diagnostics=true` 또는 `output.save_diagnostics=true`

저장 항목:
- `ess_trace`
- `resample_steps`
- `theta_trace_norm`
- `logw_mean_trace`, `logw_std_trace`, `logw_min_trace`, `logw_max_trace`
- `final_ess`, `resample_count`

저장 위치:
- `results.json` 내 `driftlite_diagnostics` (요약 직렬화)
- `data/driftlite_diagnostics.npz` (배열 원본)

## 7. Practical Commands

```bash
# 빠른 smoke test
python scripts/run_eval.py \
  eval.device=cpu \
  eval.models='[non-guided,guided,tdp,driftlite]' \
  eval.n_base=256 eval.n_guided=64 eval.bootstrap_reps=5 diffusion.n_steps=16

# DriftLite만 상세 실행
python scripts/run_model.py \
  model.name=driftlite \
  eval.device=cpu eval.n_base=512 eval.n_guided=128 diffusion.n_steps=32 \
  driftlite.n_particles=256 driftlite.resample=true driftlite.reward_path=linear \
  output.save_diagnostics=true
```
