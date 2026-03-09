# KUKA Planner Simple

`tdp-kuka` 기반 핵심 모듈을 벤더링해, 코드베이스 외부 경로 의존 없이 zero-shot 평가 실험을 실행하는 프레임워크입니다.

## 핵심 특징
- diffusion/denoising/gym_stacking 코드는 `src/` 내부에 벤더링
- 기본 체크포인트는 `assets/checkpoints/`에 포함
- 체크포인트 경로는 모델별 `diffusion_log_dir`, `guide_ckpt`에서 직접 지정
- `OmegaConf dotlist` 기반 실행
- 지원 태스크: `conditional_stack`, `pick2put`, `pnwp`
- 기본 비교 모델: `diffusion`, `tdp`
- 결과 저장:
  - `results.json`
  - `data/summary.csv`
  - `data/episode_metrics.csv`
  - `artifacts/<task>/<model>/episode_xxxx/{rollout.json,trajectory.npy,video.mp4}`

## 실행
필수 패키지(최소):
```bash
pip install -r requirements.txt
```

```bash
cd /workspace/kuka_planner_simple
python scripts/run_kuka_eval.py task=conditional_stack
python scripts/run_kuka_eval.py task=pick2put
python scripts/run_kuka_eval.py task=pnwp
```

## 빠른 스모크 테스트
```bash
python scripts/run_kuka_eval.py task=conditional_stack eval.episodes=2 eval.batch_size=8 eval.save_render=false
python scripts/run_kuka_eval.py task=pick2put eval.episodes=2 eval.batch_size=8 eval.save_render=false
python scripts/run_kuka_eval.py task=pnwp eval.episodes=2 eval.batch_size=8 eval.save_render=false
```

## 유틸리티
```bash
python scripts/list_models.py
python scripts/validate_config.py task=conditional_stack eval.episodes=2
```

## 새 모델 추가
`configs/tasks/*.yaml`의 `models` 항목에 새 엔트리를 추가하고 `planner`를 지정하면 됩니다.
`planner_params.selection_mode`와 `planner_params.pg`는 필수입니다.

예시:
```yaml
models:
  - name: my_new_model
    enabled: true
    planner: tdp
    diffusion_module: diffusion.denoising_diffusion_pytorch
    diffusion_log_dir: logs/my_model_log_dir
    diffusion_epoch: 700
    guide_ckpt: logs/my_guide/value/state_80.pt
    guide_arch:
      time_dim: 128
      input_dim: 39
      hidden_dims: [128, 128, 128]
      output_dim: 12
    planner_params:
      selection_mode: subtree   # direct|subtree
      batch_size: 64
      pg: true
      pg_scale: 0.4
```

그리고 실행 시
```bash
python scripts/run_kuka_eval.py task=conditional_stack eval.models='[my_new_model,diffusion,tdp]'
```
