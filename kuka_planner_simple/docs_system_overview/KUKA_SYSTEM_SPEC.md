# KUKA Planner Simple 시스템 명세 (최신)

이 문서는 `/workspace/kuka_planner_simple`의 현재 구조를 기준으로,
실험 흐름/역할 분리/파라미터 체계를 정리한다.

## 1. 목적

- `tdp-kuka` 기반 핵심 모듈(`diffusion`, `denoising_diffusion_pytorch`, `gym_stacking`)을 `src/`에 벤더링해 사용
- 기본 체크포인트를 `assets/checkpoints/`에 포함해 외부 repo 경로 의존 제거
- 태스크별 어댑터 + 플래너별 stage planning 전략으로 실험 수행
- 결과를 JSON/CSV/아티팩트로 일관 저장

## 2. 폴더 역할

- `configs/`: 기본/태스크/모델 설정
- `scripts/`: CLI 엔트리포인트
- `src/kuka_planner_simple/`: 실제 런타임/플래너/태스크 코드
- `outputs/`: 실행 결과
- `docs_system_overview/`: 시스템 문서

## 3. 실행 흐름

1. `scripts/run_kuka_eval.py` 실행
2. `run_eval_from_cli()` -> `load_cfg()`
3. `build_runtime()`에서 Runtime/SharedRuntime 구성
4. `run_eval_pipeline()`에서 모델별 runner 생성
5. 각 runner가 episode 반복 실행
6. `results.json`, `summary.csv`, `episode_metrics.csv`, artifacts 저장

### 3.1 구조도(요약)

`CLI(task=...)`
-> `load_cfg()`
-> `build_runtime()`
-> `run_eval_pipeline()`
-> `MODEL_REGISTRY[planner]`로 runner 생성
-> `runner.run_episode()`
-> `for stage in task_adapter.num_stages(): plan_stage -> execute -> next state`
-> 결과 저장

### 3.2 Loop 성격

- 현재 실행은 **stage 단위 closed-loop**다.
- 각 stage에서 trajectory(호라이즌)를 선택해 실행한 뒤, 실제 next state를 다시 읽어서 다음 stage를 재계획한다.
- 즉 stage 내부는 open-loop 실행, stage 간에는 closed-loop 재계획 구조다.

### 3.3 DFS 관점 순회

- DFS로 보면 방문 순서는 아래와 같다.
  1. `run_eval_pipeline`
  2. `build_runtime` 서브트리 전체 방문 후 복귀
  3. `runner 생성` 노드 방문
  4. `model -> episode -> stage` 순으로 깊게 진입
  5. stage 내부에서 `plan_stage`의 `selection_mode` 분기(`direct/subtree`) 리프까지 방문
  6. `execute` 후 상위(stage/episode/model)로 백트래킹
  7. 모든 형제 노드를 방문하면 집계/저장 노드 방문 후 종료

## 4. 설정 체계

## 4.1 Structured config (`config/spec.py`)

- `RuntimeSpec`: device/seed
- `TaskSpec`: name/horizon/diffusion_steps/diffusion_module/env_variant/value_rule
- `EvalSpec`: episodes/batch_size/save_render/save_trajectories/models
  - 기본값: `models=[diffusion, tdp]`
- `TDPSpec`: subtree refinement 계산에 쓰이는 하이퍼파라미터
- `PGSpec`: PG 기본값
- `GuideArchSpec`: guide 모델 차원
- `PlannerParamsSpec`: 모델별 planner 파라미터(단일 진실 원천)
  - `selection_mode: direct|subtree`
  - `batch_size`, `pg`, `pg_scale`, `guide_step`
  - `sample_ub`, `sample_jump_size`, `diffusion_step`
- `ModelSpec`: planner 종류 + diffusion/guide 경로 + `planner_params`
- `OutputSpec`: 출력 경로/실험명/타임존

## 4.2 병합 규칙 (`experiment_runtime.load_cfg`)

- `task=`를 special key로 분리
- 병합 순서:
  - structured defaults
  - `configs/base.yaml`
  - `configs/tasks/<task>.yaml`
  - `configs/models/default_compare.yaml`
  - dotlist

## 5. 코드 책임 분리

## 5.1 Pipelines

- `pipelines/experiment_runtime.py`
  - config 검증/병합
  - `task.diffusion_module`를 모델 기본값으로 적용
  - `planner_params` 키/모드 호환성 fail-fast 검증
  - `planner_params.pg` 명시(true/false) 강제
  - deprecated `tdp.use_tree/use_sub_tree` 차단
  - 런타임 의존성 체크
  - dataset/task adapter/shared runtime 구성
- `pipelines/run_eval_pipeline.py`
  - 모델별 runner 생성
  - episode 반복 실행
  - 집계/저장

## 5.2 Planners

- `planners/base.py`
  - `EpisodeResult`, `PlannerRunner` 프로토콜
- `planners/registry.py`
  - `SharedRuntime` 정의
  - `MODEL_REGISTRY` 정의
  - `KukaRunner` (공통 오케스트레이션)
    - env reset
    - stage loop
    - artifact 저장
- `planners/stage_planner.py`
  - `StagePlanner`: stage 단위 planning 전략 담당
  - `ResolvedPlannerParams`: 실행 시점 확정 파라미터

### 핵심 설계

- Runner는 "실행 루프 + 저장"만 담당
- StagePlanner는 "후보 생성/스코어/선택" 담당
- TaskAdapter는 "task-specific 컨텍스트/실행" 담당

## 5.3 Tasks

- `tasks/adapters.py`
  - `BaseTaskAdapter` 인터페이스
  - `ConditionalStackAdapter`, `Pick2PutAdapter`, `PnwpAdapter`
  - `TASK_ADAPTERS` 매핑 + `supported_task_names()`로 지원 task 단일 소스 관리
  - `StageContext` 정의

## 6. StagePlanner 로직

- `resolve_params()`
  - `planner_params` + 전역 기본값(`cfg.tdp`, `cfg.pg`)으로 최종 파라미터 확정
  - `selection_mode` 및 `pg` 필수 검증 (`direct|subtree`, `pg=true|false`)
- `plan_stage()`
  - `guided_sample`로 후보 trajectory 생성
  - `compute_values`로 점수 계산
  - 분기:
    - `direct`: best sample 선택
    - `subtree`: fast-guided sub-sample 생성 후 main/sub 중 최고값 선택

### 6.1 PG(TDP-PG) 적용 지점

- PG는 `StagePlanner._guided_candidates()`에서 task adapter의 `guided_sample()` 호출 시 적용된다.
- `pg`, `pg_scale`, `guide_step`은 `resolve_params()`에서 결정된다.
- planner별 기본 성향:
  - `diffusion`: direct + PG 기본 비활성
  - `tdp`: direct/subtree + PG 기본 활성

### 6.2 DP/TDP 구분 방식

- 별도 거대한 runner를 나누지 않고, 동일 실행 뼈대에서 `selection_mode`와 PG 여부로 동작을 분기한다.
- 핵심 토글은 모델별 `planner_params.selection_mode`다.

## 7. 아티팩트 명세

각 episode 디렉토리:

- `rollout.json`
  - `task`, `model`, `planner`, `planning_mode`
  - `resolved_params`, `model_config`
  - `score`, `success`, `elapsed_sec`, `seed`
- `trajectory.npy` (`save_trajectories=true`)
- `video.mp4` (`save_render=true`)

상위 결과:

- `results.json`
- `data/summary.csv`
- `data/episode_metrics.csv`

## 8. 확장 포인트

- 새 planner:
  - `MODEL_REGISTRY`에 factory 등록
  - 필요 시 custom runner 또는 custom stage planner 사용
- 새 task:
  - `BaseTaskAdapter` 구현 추가
  - `make_task_adapter` 분기 추가

## 9. 운영 체크포인트

- 체크포인트/로그 경로는 모델별 `diffusion_log_dir`, `guide_ckpt`에 직접 지정
- pybullet/gym/easydict/imageio 설치
- 체크포인트 경로(`diffusion_log_dir`, `guide_ckpt`) 확인
- 스모크 테스트는 `eval.episodes=1~2`, 작은 `batch_size`로 먼저 수행

## 10. 자주 묻는 질문

- Q. `num_stages`는 무엇인가?
  - A. 한 episode를 몇 번의 `plan -> execute` 사이클로 나눌지 task adapter가 정의한 값이다.
- Q. 예측은 horizon 단위인가?
  - A. 네. stage마다 horizon trajectory 후보를 만들고 하나를 선택해 실행한다.
- Q. `runners = { ... }`는 무슨 의미인가?
  - A. 선택된 모델 목록을 순회하며 planner registry에서 runner 인스턴스를 생성하는 코드다.
