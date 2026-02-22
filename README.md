# Toy Diffuser Simple

Closed-form diffusion toy 실험 코드입니다. 
세 가지 샘플러(`non-guided`, `guided`, `tdp`)를 같은 보상 함수에서 비교하고, 
SNIS로 추정한 tilted target과 통계적으로 얼마나 가까운지 평가합니다.

## 무엇을 할 수 있나
1. 단일 모델 성능 비교 (`run_model.py`)
2. 여러 모델 일괄 비교 (`run_eval.py`)
3. 샘플 분포/경로 시각화 (`visualize.py`)

## 빠른 시작

```bash
conda env create -f environment.yml
conda activate toy_diffuser
```

## 실행 파일

### `scripts/run_model.py`
모델 하나만 실행하고 tilted target과 비교합니다.

```bash
python scripts/run_model.py eval.device=cpu
python scripts/run_model.py model.name=tdp eval.device=cpu
```

기본값
- `model.name=guided`

출력
- `outputs/run_model/YYYY-MM-DD/HH-MM-SS/`
- 파일: `config.yaml`, `results.json`

### `scripts/run_eval.py`
모델 목록을 한 번에 실행합니다.

```bash
python scripts/run_eval.py eval.device=cpu
python scripts/run_eval.py eval.device=cpu eval.models='[non-guided,guided,tdp]'
```

기본값
- `eval.models`: 등록된 모든 모델 (`non-guided`, `guided`, `tdp`)

출력
- `outputs/run_eval/YYYY-MM-DD/HH-MM-SS/`
- 파일: `config.yaml`, `results.json`, `data/summary.csv`, `data/derived_saved_subsets.npz`, `plots/*.png`

### `scripts/visualize.py`
모델 샘플을 시각화합니다.

```bash
python scripts/visualize.py eval.device=cpu
python scripts/visualize.py eval.device=cpu viz.models='[guided,tdp]' viz.path_model=tdp viz.n_traj=12
```

기본값
- `viz.models`: 등록된 모든 모델
- `viz.path_model=guided`
- `viz.n_traj=5`

출력
- `outputs/visualize/YYYY-MM-DD/HH-MM-SS/`
- 파일: `config.yaml`, `plots/final_states_scatter.png`, `plots/reward_hist.png`, `plots/traj_paths.png`

## 옵션(의미 + 기본값)

### Trajectory (`traj.*`)
| 옵션 | 기본값 | 의미 | 언제 바꾸나 |
|---|---:|---|---|
| `traj.horizon_T` | `32` | trajectory 길이 T | 더 긴 horizon 실험 |
| `traj.action_dim` | `2` | action/state 차원 | 차원 민감도 실험 |
| `traj.base_action_mean` | `0.1` | base GMM 평균 크기 | 난이도/모드 분리 조절 |
| `traj.sigma0_sq` | `0.025` | base 분산(등방) | base 다양성 조절 |
| `traj.pi_pos` | `0.5` | positive mixture 비율 | 불균형 mixture 실험 |

### Reward (`reward.*`)
| 옵션 | 기본값 | 의미 | 언제 바꾸나 |
|---|---:|---|---|
| `reward.w_neg` | `0.2` | negative goal 가중치 | 반대 goal 페널티 조절 |
| `reward.w_pos` | `0.7` | positive goal 가중치 | 목표 집중도 조절 |
| `reward.offset` | `0.1` | 보상 상수항 | 보상 baseline 조정 |
| `reward.state_var` | `0.25` | 상태 가우시안 분산 | goal 주변 폭 조절 |
| `reward.goal_x` | `None` | goal x 좌표(2D 편의) | 목표 위치 수동 지정 |
| `reward.goal_y` | `None` | goal y 좌표(2D 편의) | 목표 위치 수동 지정 |
| `reward.goal` | `None` | goal 벡터(차원 일반화) | 2D 외 차원에서 goal 지정 |

`reward.goal`이 주어지면 `goal_x/y`보다 우선합니다. 둘 다 없으면 내부 기본 goal을 사용합니다.

### Diffusion (`diffusion.*`)
| 옵션 | 기본값 | 의미 | 언제 바꾸나 |
|---|---:|---|---|
| `diffusion.n_steps` | `256` | diffusion step 수 | 정확도/시간 trade-off |
| `diffusion.beta_start` | `1e-4` | beta 시작값 | 노이즈 스케줄 실험 |
| `diffusion.beta_end` | `2e-2` | beta 종료값 | 노이즈 스케줄 실험 |

### Guidance (`guidance.*`)
| 옵션 | 기본값 | 의미 | 언제 바꾸나 |
|---|---:|---|---|
| `guidance.enabled` | `true` | guidance on/off | guided vs pure reverse 비교 |
| `guidance.scale` | `10.0` | guidance 강도 | 성능/편향 강도 조절 |
| `guidance.clip_norm` | `1.0` | gradient clip norm | 큰 gradient 안정화 |

### Evaluation (`eval.*`)
| 옵션 | 기본값 | 의미 | 언제 바꾸나 |
|---|---:|---|---|
| `eval.seed` | `0` | 랜덤 시드 | 재현성 |
| `eval.device` | `auto` | 실행 디바이스 | CPU/GPU 강제 선택 |
| `eval.n_base` | `50000` | SNIS base 샘플 수 | tilted 추정 안정화 |
| `eval.n_guided` | `20000` | 모델 샘플 수 | 통계 오차 감소 |
| `eval.bootstrap_reps` | `200` | 부트스트랩 반복 수 | SE 추정 안정화 |
| `eval.batch_size` | `131072` | 샘플링 배치 크기 | 메모리/속도 조절 |
| `eval.f_list` | `['final_x','final_y','pos_indicator','R']` | 비교 지표 목록 | 원하는 지표만 평가 |
| `eval.models` | `[]` | 실행 모델 목록 | `run_eval.py`에서 모델 subset 실행 |

`eval.models=[]`이면 등록된 모든 모델을 실행합니다.

### TDP (`tdp.*`)
| 옵션 | 기본값 | 의미 | 언제 바꾸나 |
|---|---:|---|---|
| `tdp.n_roots` | `64` | parent 수 B | 탐색 폭 조절 |
| `tdp.renoise_frac` | `0.15` | 재노이즈 step 비율(0~1) | 변형 강도 고정 조절 |
| `tdp.topk_final` | `1` | rollout당 반환 elite 개수 | 다양성/선택 폭 조절 |

### Output (`output.*`)
| 옵션 | 기본값 | 의미 | 언제 바꾸나 |
|---|---:|---|---|
| `output.root` | `outputs` | 결과 저장 루트 | 저장 경로 변경 |
| `output.timezone` | `Asia/Seoul` | 타임스탬프 시간대 | 팀 표준 시간대 맞춤 |
| `output.save_base_max` | `50000` | 저장할 base 최대 개수 | 용량 절감 |
| `output.save_chain_max` | `20000` | 모델별 저장 최대 개수 | 용량 절감 |
| `output.save_plots_max_points` | `50000` | 플롯 최대 포인트 수 | 렌더링/용량 절감 |

## 현재 TDP 동작

현재 `tdp`는 아래 절차로 동작합니다.
1. parent `B`개 생성
2. parent마다 `u ~ Uniform[0, horizon_T)` timestep index를 뽑아 해당 timestep action block extract
3. `tdp.renoise_frac` 비율(step)로 재노이즈
4. 디노이징 중 매 step extract timestep block overwrite
5. parent+child(`2B`) 중 reward 상위 `topk_final` 선택

`tdp`는 항상 guided 모드로 실행됩니다(별도 on/off 옵션 없음).

구현 파일: `src/models/tdp.py`

## 자주 쓰는 커맨드

```bash
# 빠른 스모크 테스트
python scripts/run_eval.py eval.device=cpu eval.n_base=200 eval.n_guided=80 eval.bootstrap_reps=10 diffusion.n_steps=16

# TDP만 비교
python scripts/run_eval.py eval.device=cpu eval.models='[tdp]' tdp.n_roots=64

# 고정밀 평가(시간 오래 걸림)
python scripts/run_eval.py eval.device=cuda eval.n_base=200000 eval.n_guided=80000 eval.bootstrap_reps=300
```

## 참고

- 시각화(`scatter`, `traj_paths`)는 현재 2D 축 표시를 가정합니다.
- CPU 환경에서도 PyTorch CUDA probe 경고가 보일 수 있으나, 보통 실행에는 영향이 없습니다.
