# Toy Diffuser (Closed-form DDPM reverse + Guidance vs Tilted)

이 레포는 **GMM base 분포 + 닫힌형(analytic) DDPM reverse kernel**을 사용해

- **Tilted target**: \(\tilde p(\tau_0)\propto p(\tau_0)\exp(R(\tau_0))\) (SNIS로 추정)
- **Non-guided reverse sampling**: \(q(\tau_{t-1}\mid \tau_t)\)만으로 샘플링
- **Guided reverse sampling**: reverse step에 \( \nabla_{\tau_t} J(\mu(\tau_t))\) 기반 mean shift 추가

이 세 분포를 비교하는 toy 실험 코드입니다.

---

## Directory Layout

```text
.
├─ src/
│  ├─ spec.py        # config dataclasses (traj/reward/diffusion/guidance/eval)
│  ├─ gmm.py         # base GMM + closed-form posteriors q(tau0|taut), q(tau_{t-1}|tau_t)
│  ├─ plan.py        # diffusion schedule + sampler (guided/non-guided)
│  ├─ reward.py      # deterministic dynamics + reward R(tau)
│  └─ eval.py        # SNIS tilted estimator + bootstrap SE + moment comparison
├─ scripts/
│  ├─ run_eval.py    # main experiment: save results to outputs/run_eval/...
│  └─ visualize.py   # visualization: save figures to outputs/visualize/...
└─ environment.yml
```

---

## Setup

### Conda
```bash
conda env create -f environment.yml
conda activate toy_diffuser
```

### GPU 0만 사용 (권장)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py eval.device=cuda
```

---

## Outputs (자동 저장 구조)

### `scripts/run_eval.py`
실행할 때마다 아래 경로가 새로 생성됩니다.

```text
outputs/run_eval/YYYY-MM-DD/HH-MM-SS/
  config.yaml
  results.json
  data/
    summary.csv
    base_tau0_*.pt
    nonguided_tau0_*.pt
    guided_tau0_*.pt
    derived_saved_subsets.npz
  plots/
    final_states_scatter.png
    reward_hist.png
```

- `results.json`: tilted(SNIS) vs non-guided vs guided 비교 결과(평균/표준오차 등)
- `data/*.pt`: 샘플 \(\tau_0\) (downsample 저장)
- `derived_saved_subsets.npz`: 저장된 subset 기준으로 \(s_T\), \(R\) 등 파생값
- `summary.csv`: 핵심 지표를 한 줄씩 정리한 CSV
- `plots/*.png`: 산점도/히스토그램

> 샘플 저장량은 `scripts/run_eval.py` 상단 `SAVE_BASE_MAX`, `SAVE_CHAIN_MAX`로 제한됩니다(디스크 폭발 방지).

### `scripts/visualize.py`
```text
outputs/visualize/YYYY-MM-DD/HH-MM-SS/
  config.yaml
  plots/
    final_states_scatter.png
    reward_hist.png
    traj_paths_guided.png
    (선택) traj_paths_nonguided.png
```

- `traj_paths_guided.png`: **한 trajectory의 \(s_t\)가 t=0→T까지 이동한 XY 경로**를 점+선으로 표시
- trajectory 개수는 `viz.n_traj`로 조절(기본 5)

---

## Run: Evaluation (`scripts/run_eval.py`)

기본:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py eval.device=cuda
```

샘플 더 많이(통계력 강화):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py eval.device=cuda \
  eval.n_base=200000 eval.n_guided=80000 eval.batch_size=4096 eval.bootstrap_reps=200
```

diffusion step 수 증가:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py eval.device=cuda diffusion.n_steps=512
```

guidance 강도 변경:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py eval.device=cuda guidance.scale=20
```

guidance 끄기(= guided와 non-guided 동일):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py eval.device=cuda guidance.enabled=false
```

---

## Run: Visualization (`scripts/visualize.py`)

기본(trajectory 5개):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/visualize.py eval.device=cuda
```

trajectory 개수 조절:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/visualize.py eval.device=cuda viz.n_traj=12
```

---

## Config / Args (조절 가능한 변수)

두 스크립트 모두 `key=value` 형태의 dotlist override를 받습니다.

### 1) Trajectory / Base distribution (`traj.*`)
- `traj.horizon_T` (int): 시계열 길이 \(T\)
- `traj.action_dim` (int): action 차원(기본 2)
- `traj.base_action_mean` (float): base GMM mean의 크기 (±m, 모든 차원에 반복)
- `traj.sigma0_sq` (float): base covariance (isotropic), 기본 \(1/40\)

예:
```bash
python scripts/run_eval.py traj.horizon_T=64 traj.base_action_mean=0.08
```

### 2) Reward (`reward.*`)
- `reward.w_neg` (float): negative goal 쪽 가우시안 weight
- `reward.w_pos` (float): positive goal 쪽 가우시안 weight
- `reward.offset` (float): 상수항
- `reward.state_var` (float): 상태 가우시안 covariance 스케일 (I * state_var)
- `reward.goal_x`, `reward.goal_y` (float, optional):
  - **없으면 자동으로** `(traj.base_action_mean*T, traj.base_action_mean*T)` 사용
  - override 주면 goal 위치만 변경

예:
```bash
python scripts/run_eval.py reward.goal_x=5.0 reward.goal_y=2.0
```

### 3) Diffusion schedule (`diffusion.*`)
- `diffusion.n_steps` (int): DDPM step 수
- `diffusion.beta_start` (float)
- `diffusion.beta_end` (float)

예:
```bash
python scripts/run_eval.py diffusion.n_steps=512 diffusion.beta_end=0.03
```

### 4) Guidance (`guidance.*`)
- `guidance.enabled` (bool): guidance on/off
- `guidance.scale` (float): mean shift 스케일
- `guidance.clip_norm` (float or null): per-sample gradient norm clipping (끄려면 `null`)

예:
```bash
python scripts/run_eval.py guidance.scale=15 guidance.clip_norm=1.0
python scripts/run_eval.py guidance.clip_norm=null
```

### 5) Evaluation (`eval.*`)
- `eval.seed` (int)
- `eval.device` (str): `auto` / `cpu` / `cuda`
- `eval.n_base` (int): base 샘플 수(tilted SNIS)
- `eval.n_guided` (int): guided/non-guided 샘플 수
- `eval.batch_size` (int): reverse sampling batch size
- `eval.bootstrap_reps` (int): bootstrap 반복 수
- `eval.f_list` (list[str]): 비교할 \(f(\tau_0)\) 목록  
  기본 지원 예시: `final_x`, `final_y`, `pos_indicator`, `R`

예:
```bash
python scripts/run_eval.py eval.f_list='[final_x,final_y,R]'
```

### 6) Visualization only (`viz.*`) — `scripts/visualize.py` 전용
- `viz.n_traj` (int): traj path로 그릴 trajectory 개수 (기본 5)

예:
```bash
python scripts/visualize.py viz.n_traj=20
```

---

## Interpreting Results

`run_eval.py`는 각 \(f\)에 대해 다음을 출력합니다.

- `tilted (SNIS) mean ± se`: \(\tilde p(\tau_0)\propto p(\tau_0)\exp(R(\tau_0))\)에서의 추정 기댓값
- `non-guided mean ± se`: non-guided reverse chain 샘플 평균
- `guided mean ± se`: guided reverse chain 샘플 평균
- 차이: `(non-guided - tilted)`, `(guided - tilted)`

차이가 유의미하게 크면(오차범위 대비 충분히 크면),
\[
p_{\text{guided}}(\tau_0)\neq \tilde p(\tau_0)
\]
를 generator-matching 관점에서 강하게 시사합니다.

---

## Practical Tips

- GPU 0 고정:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py eval.device=cuda
  ```
- 메모리 부족하면:
  - `eval.batch_size` 줄이기
  - `eval.n_guided`, `eval.n_base` 줄이기
- 디스크 용량 걱정이면:
  - `scripts/run_eval.py` 상단 `SAVE_BASE_MAX`, `SAVE_CHAIN_MAX`를 줄이기
