# CenterPoint dist_test.py VoxelNet 추론 흐름 분석

## 1. 전체 요약

- `tools/dist_test.py`는 CLI 인자로 받은 config를 `Config.fromfile()`로 읽고, `cfg.model`을 `build_detector()`에 넘겨 모델을 만든다.
- `cfg.model.type == "VoxelNet"`이면 `DETECTORS` registry에서 `"VoxelNet"` 문자열로 `VoxelNet` 클래스를 찾아 생성한다.
- `VoxelNet.__init__()`은 부모인 `SingleStageDetector.__init__()`로 reader, backbone, neck, bbox_head config를 넘기고, 각 모듈은 별도 registry에서 생성된다.
- 테스트 루프에서는 `data_loader`가 만든 `data_batch`를 `batch_processor(..., train_mode=False)`에 넘기고, 내부에서 `model(example, return_loss=False)`를 호출한다.
- `VoxelNet.forward(return_loss=False)`는 feature 추출, bbox head 예측, `bbox_head.predict()` 후처리를 거쳐 detection 결과를 만들고, `prediction.pkl` 저장 및 `dataset.evaluation()`으로 이어진다.

대표 config 기준:

| 항목 | 파일:라인 | 값 |
|---|---:|---|
| detector type | `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:23-25` | `model = dict(type="VoxelNet", ...)` |
| reader | `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:26-30` | `VoxelFeatureExtractorV3` |
| backbone | `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:31-33` | `SpMiddleResNetFHD` |
| neck | `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:34-43` | `RPN` |
| bbox_head | `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:44-54` | `CenterHead` |
| val/test dataset | `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:170-202` | `NuScenesDataset` |

## 2. 핵심 호출 트리

```text
tools/dist_test.py:main()
├─ Line 80: args = parse_args()
├─ Line 82: cfg = Config.fromfile(args.config)
│  └─ det3d/torchie/utils/config.py:78 Config.fromfile()
│     └─ Line 100: return Config(cfg_dict, filename=filename)
├─ Line 106: model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
│  └─ det3d/models/builder.py:49 build_detector()
│     └─ Line 50: build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
│        └─ det3d/models/builder.py:16 build()
│           └─ Line 21: build_from_cfg(cfg, registry, default_args)
│              └─ det3d/utils/registry.py:49 build_from_cfg()
│                 ├─ Line 61: obj_type = args.pop("type")
│                 ├─ Line 63: obj_cls = registry.get(obj_type)
│                 └─ Line 78: return obj_cls(**args)
│                    └─ det3d/models/detectors/voxelnet.py:8 VoxelNet
│                       └─ Line 19: super(VoxelNet, self).__init__(...)
│                          └─ det3d/models/detectors/single_stage.py:12 SingleStageDetector.__init__()
│                             ├─ Line 23: self.reader = builder.build_reader(reader)
│                             ├─ Line 24: self.backbone = builder.build_backbone(backbone)
│                             ├─ Line 26: self.neck = builder.build_neck(neck)
│                             └─ Line 27: self.bbox_head = builder.build_head(bbox_head)
├─ Line 110/113: dataset = build_dataset(cfg.data.test or cfg.data.val)
│  └─ det3d/datasets/builder.py:31 build_dataset()
│     └─ Line 41: build_from_cfg(cfg, DATASETS, default_args)
├─ Line 115: data_loader = build_dataloader(...)
│  └─ det3d/datasets/loader/build_loader.py:23 build_dataloader()
│     └─ Line 46: DataLoader(...)
├─ Line 123: checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
├─ Line 157: for i, data_batch in enumerate(data_loader):
│  └─ Line 167: outputs = batch_processor(model, data_batch, train_mode=False, ...)
│     └─ det3d/torchie/apis/train.py:92 batch_processor()
│        ├─ Line 100: example = example_to_device(data, device, non_blocking=False)
│        └─ Line 113: return model(example, return_loss=False)
│           └─ det3d/models/detectors/voxelnet.py:55 VoxelNet.forward()
│              ├─ Line 56: x, _ = self.extract_feat(example)
│              │  └─ det3d/models/detectors/voxelnet.py:23 VoxelNet.extract_feat()
│              │     ├─ Line 44: input_features = self.reader(...)
│              │     ├─ Line 46: x, voxel_feature = self.backbone(...)
│              │     └─ Line 51: x = self.neck(x)
│              ├─ Line 57: preds, _ = self.bbox_head(x)
│              └─ Line 62: return self.bbox_head.predict(example, preds, self.test_cfg)
├─ Line 177: detections.update({token: output})
├─ Line 185: all_predictions = all_gather(detections)
├─ Line 199: save_pred(predictions, args.work_dir)
└─ Line 201: dataset.evaluation(copy.deepcopy(predictions), ...)
```

## 3. dist_test.py main() 상세 흐름

| 단계 | 파일:라인 | 코드/함수 | 역할 | 다음으로 이어지는 곳 |
|---:|---|---|---|---|
| 1 | `tools/dist_test.py:80` | `args = parse_args()` | config, work_dir, checkpoint, distributed 옵션을 CLI에서 읽는다. | `Config.fromfile()` |
| 2 | `tools/dist_test.py:82` | `cfg = Config.fromfile(args.config)` | config 파일을 Python/YAML/JSON에서 읽어 `Config` 객체로 만든다. | `cfg.model`, `cfg.data`, `cfg.test_cfg` 사용 |
| 3 | `tools/dist_test.py:83` | `cfg.local_rank = args.local_rank` | 현재 프로세스 GPU rank를 cfg에 저장한다. | distributed 설정 |
| 4 | `tools/dist_test.py:86-87` | `cfg.work_dir = args.work_dir` | CLI의 저장 경로가 config보다 우선한다. | logger, prediction 저장 |
| 5 | `tools/dist_test.py:89-99` | `distributed`, `cfg.gpus` 설정 | `WORLD_SIZE`에 따라 단일/분산 테스트를 결정한다. | GPU 배치 |
| 6 | `tools/dist_test.py:102` | `get_root_logger(cfg.log_level)` | config의 log level로 logger 생성. | 실행 로그 |
| 7 | `tools/dist_test.py:106` | `build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)` | cfg의 model dict로 detector 객체 생성. | `VoxelNet` 생성 |
| 8 | `tools/dist_test.py:108-113` | `build_dataset(cfg.data.test/val)` | `--testset` 여부에 따라 test 또는 val dataset 생성. | `build_dataloader()` |
| 9 | `tools/dist_test.py:115-121` | `build_dataloader(...)` | batch size, worker, dist, shuffle 옵션으로 DataLoader 생성. | 테스트 루프 |
| 10 | `tools/dist_test.py:123` | `load_checkpoint(model, args.checkpoint, map_location="cpu")` | checkpoint를 model에 로드한다. | GPU 이동 |
| 11 | `tools/dist_test.py:126-137` | `DistributedDataParallel(...)` 또는 `model.cuda()` | 모델을 GPU에 올린다. 분산이면 DDP wrapper로 감싼다. | `model.eval()` |
| 12 | `tools/dist_test.py:139` | `model.eval()` | 평가 모드로 전환한다. | `for data_batch` |
| 13 | `tools/dist_test.py:157` | `for i, data_batch in enumerate(data_loader):` | DataLoader에서 batch 단위 입력을 받는다. | `batch_processor()` |
| 14 | `tools/dist_test.py:166-169` | `batch_processor(..., train_mode=False, ...)` | no_grad 추론 실행. | `VoxelNet.forward(return_loss=False)` |
| 15 | `tools/dist_test.py:170-179` | `token`, `detections.update(...)` | 각 output을 sample token 기준으로 저장한다. | `all_gather()` |
| 16 | `tools/dist_test.py:183-185` | `synchronize()`, `all_gather(detections)` | 분산 프로세스 간 결과를 모은다. | rank 0 집계 |
| 17 | `tools/dist_test.py:192-199` | `predictions.update(...)`, `save_pred(...)` | 모든 prediction을 합치고 `prediction.pkl`로 저장한다. | evaluation |
| 18 | `tools/dist_test.py:201` | `dataset.evaluation(...)` | dataset별 평가 함수를 호출한다. | metric 출력 |

## 4. cfg.model에서 VoxelNet 객체가 생성되는 과정

| 단계 | 파일:라인 | 호출 | 입력 | 출력 | 설명 |
|---:|---|---|---|---|---|
| 1 | `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:23-25` | `model = dict(type="VoxelNet", ...)` | config 파일 | `cfg.model.type == "VoxelNet"` | config 안의 문자열이 detector 종류를 지정한다. |
| 2 | `tools/dist_test.py:106` | `build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)` | `cfg.model`, `cfg.test_cfg` | detector 객체 | 테스트용 model 생성 진입점. |
| 3 | `det3d/models/builder.py:49-50` | `build_detector()` | `cfg`, `train_cfg`, `test_cfg` | `build(...)` 반환값 | `DETECTORS` registry와 default args를 넘긴다. |
| 4 | `det3d/models/builder.py:16-21` | `build(cfg, DETECTORS, ...)` | model config, detector registry | `build_from_cfg(...)` 반환값 | cfg가 list가 아니면 registry 기반 생성으로 넘어간다. |
| 5 | `det3d/utils/registry.py:60-63` | `obj_type = args.pop("type")`, `registry.get(obj_type)` | `"VoxelNet"` | `VoxelNet` 클래스 | 문자열 type으로 registry에서 class를 찾는다. |
| 6 | `det3d/utils/registry.py:74-78` | `args.setdefault(...)`, `return obj_cls(**args)` | reader/backbone/neck/head config + `train_cfg=None`, `test_cfg=cfg.test_cfg` | `VoxelNet(...)` instance | config dict가 constructor keyword argument가 된다. |
| 7 | `det3d/models/detectors/voxelnet.py:7-8` | `@DETECTORS.register_module`, `class VoxelNet(...)` | class object | registry 등록 | import 시 `VoxelNet`이라는 이름으로 `DETECTORS`에 등록된다. |
| 8 | `det3d/utils/registry.py:44-46` | `register_module()` | `VoxelNet` class | 등록된 class | decorator가 `_register_module()` 호출 후 class를 그대로 반환한다. |
| 9 | `det3d/utils/registry.py:37-42` | `_register_module()` | class name `"VoxelNet"` | `_module_dict["VoxelNet"] = VoxelNet` | registry 내부 dict에 class를 저장한다. |
| 10 | `det3d/models/detectors/voxelnet.py:9-21` | `VoxelNet.__init__()` | reader/backbone/neck/bbox_head config | initialized detector | 바로 부모 `SingleStageDetector.__init__()`을 호출한다. |
| 11 | `det3d/models/detectors/single_stage.py:12-31` | `SingleStageDetector.__init__()` | 각 submodule config | reader/backbone/neck/head 포함 model | 실제 submodule 객체들이 여기서 생성된다. |

관련 import 흐름:

| 파일:라인 | 역할 |
|---|---|
| `det3d/models/__init__.py:8-19` | bbox head, builder, detectors, necks, readers 등을 import하여 registry 등록 부작용을 발생시킨다. |
| `det3d/models/detectors/__init__.py:1-5` | `VoxelNet` 포함 detector class들을 import한다. |
| `det3d/models/registry.py:3-10` | `READERS`, `BACKBONES`, `NECKS`, `HEADS`, `DETECTORS` registry 객체를 만든다. |

## 5. reader / backbone / neck / bbox_head 생성 흐름

대표 config `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py` 기준이다. 다른 VoxelNet config도 `type` 문자열만 다를 수 있고 생성 방식은 동일하다.

| 모듈 | cfg 경로 | builder 함수 | registry | 최종 객체 | 역할 |
|---|---|---|---|---|---|
| reader | `cfg.model.reader` (`configs/...fix_bn_z.py:26-30`) | `det3d/models/builder.py:30 build_reader()` | `READERS` (`det3d/models/registry.py:3`) | `VoxelFeatureExtractorV3` (`det3d/models/readers/voxel_encoder.py:8-9`) | voxel feature를 point 평균 feature로 압축한다. |
| backbone | `cfg.model.backbone` (`configs/...fix_bn_z.py:31-33`) | `det3d/models/builder.py:34 build_backbone()` | `BACKBONES` (`det3d/models/registry.py:4`) | `SpMiddleResNetFHD` (`det3d/models/backbones/scn.py:97-98`) | sparse convolution으로 BEV feature map을 만든다. |
| neck | `cfg.model.neck` (`configs/...fix_bn_z.py:34-43`) | `det3d/models/builder.py:38 build_neck()` | `NECKS` (`det3d/models/registry.py:5`) | `RPN` (`det3d/models/necks/rpn.py:22-23`) | multi-scale BEV feature를 upsample/concat한다. |
| bbox_head | `cfg.model.bbox_head` (`configs/...fix_bn_z.py:44-54`) | `det3d/models/builder.py:41 build_head()` | `HEADS` (`det3d/models/registry.py:6`) | `CenterHead` (`det3d/models/bbox_heads/center_head.py:166-167`) | heatmap/regression head 출력과 decode/NMS 후처리를 담당한다. |

생성 위치:

| 파일:라인 | 코드 | 입력 | 출력 |
|---|---|---|---|
| `det3d/models/detectors/single_stage.py:23` | `self.reader = builder.build_reader(reader)` | `cfg.model.reader` | reader 객체 |
| `det3d/models/detectors/single_stage.py:24` | `self.backbone = builder.build_backbone(backbone)` | `cfg.model.backbone` | backbone 객체 |
| `det3d/models/detectors/single_stage.py:25-26` | `if neck is not None: self.neck = builder.build_neck(neck)` | `cfg.model.neck` | neck 객체 |
| `det3d/models/detectors/single_stage.py:27` | `self.bbox_head = builder.build_head(bbox_head)` | `cfg.model.bbox_head` | bbox head 객체 |
| `det3d/models/detectors/single_stage.py:28-29` | `self.train_cfg`, `self.test_cfg` 저장 | `train_cfg=None`, `test_cfg=cfg.test_cfg` | forward에서 사용될 cfg |

## 6. 테스트 루프와 batch_processor 흐름

| 단계 | 파일:라인 | 호출 | 입력 | 출력 | 설명 |
|---:|---|---|---|---|---|
| 1 | `tools/dist_test.py:157` | `for i, data_batch in enumerate(data_loader):` | `DataLoader` | `data_batch` | dataset pipeline과 collate를 거친 batch가 들어온다. |
| 2 | `tools/dist_test.py:166` | `with torch.no_grad():` | 없음 | gradient 비활성화 context | 추론이므로 autograd 저장을 막는다. |
| 3 | `tools/dist_test.py:167-169` | `batch_processor(model, data_batch, train_mode=False, local_rank=args.local_rank)` | model, batch, `False`, rank | `outputs` | train 모드가 아니라 예측 결과 list를 기대한다. |
| 4 | `det3d/torchie/apis/train.py:92-97` | `batch_processor()` device 선택 | `local_rank` | `torch.device(local_rank)` | GPU rank를 device로 사용한다. |
| 5 | `det3d/torchie/apis/train.py:100` | `example = example_to_device(data, device, non_blocking=False)` | batch dict | GPU로 이동한 `example` | Tensor/list Tensor를 device로 옮긴다. |
| 6 | `det3d/torchie/apis/train.py:28-57` | `example_to_device()` | `example` dict | `example_torch` dict | `voxels`, `coordinates`, `num_points`, `points` 등 key별로 이동 방식이 다르다. |
| 7 | `det3d/torchie/apis/train.py:104-113` | `if train_mode ... else return model(example, return_loss=False)` | `train_mode=False` | model output | 추론 branch로만 간다. |
| 8 | `det3d/models/detectors/voxelnet.py:55-62` | `VoxelNet.forward(example, return_loss=False)` | `example` | detection list | loss 대신 `bbox_head.predict()`를 반환한다. |

DataLoader 생성 근거:

| 파일:라인 | 코드/함수 | 설명 |
|---|---|---|
| `tools/dist_test.py:115-121` | `build_dataloader(dataset, batch_size=..., workers_per_gpu=..., dist=distributed, shuffle=False)` | 테스트용 DataLoader 생성. |
| `det3d/datasets/loader/build_loader.py:23-25` | `def build_dataloader(...)` | DataLoader helper 진입점. |
| `det3d/datasets/loader/build_loader.py:27-43` | sampler, batch_size, num_workers 설정 | distributed 여부와 shuffle에 따라 sampler 결정. |
| `det3d/datasets/loader/build_loader.py:46-55` | `DataLoader(..., collate_fn=collate_kitti, pin_memory=False)` | PyTorch DataLoader를 실제 생성한다. |

## 7. VoxelNet.forward(return_loss=False) 상세 흐름

```text
det3d/models/detectors/voxelnet.py:55 VoxelNet.forward(example, return_loss=False)
├─ Line 56: x, _ = self.extract_feat(example)
│  └─ det3d/models/detectors/voxelnet.py:23 extract_feat(data)
│     ├─ Line 24: if 'voxels' not in data:
│     │  ├─ Line 25: output = self.reader(data['points'])
│     │  └─ Line 28-35: data dict 재구성, input_features = voxels
│     ├─ Line 36: else:
│     │  ├─ Line 37-43: 기존 voxel batch에서 features/num_voxels/coors/input_shape 구성
│     │  └─ Line 44: input_features = self.reader(data["features"], data['num_voxels'])
│     ├─ Line 46-48: x, voxel_feature = self.backbone(input_features, coors, batch_size, input_shape)
│     ├─ Line 50-51: if self.with_neck: x = self.neck(x)
│     └─ Line 53: return x, voxel_feature
├─ Line 57: preds, _ = self.bbox_head(x)
├─ Line 59-60: if return_loss: return loss(...)
└─ Line 61-62: else: return self.bbox_head.predict(example, preds, self.test_cfg)
```

| 단계 | 파일:라인 | 코드 | 입력 | 출력 | 설명 |
|---:|---|---|---|---|---|
| 1 | `det3d/models/detectors/voxelnet.py:55` | `def forward(self, example, return_loss=True, **kwargs)` | `example`, `return_loss=False` | detection list | batch_processor에서 호출되는 모델 forward. |
| 2 | `det3d/models/detectors/voxelnet.py:56` | `x, _ = self.extract_feat(example)` | GPU로 옮겨진 example | BEV feature `x` | reader/backbone/neck feature 추출로 들어간다. |
| 3 | `det3d/models/detectors/voxelnet.py:23-25` | `if 'voxels' not in data: self.reader(data['points'])` | raw points case | `voxels, coors, shape` | raw points에서 dynamic voxel을 만드는 branch. |
| 4 | `det3d/models/detectors/voxelnet.py:36-44` | `else: ... self.reader(data["features"], data['num_voxels'])` | pre-voxelized batch | `input_features` | 일반 dataset batch에 `voxels`가 있으면 이 branch가 쓰인다. |
| 5 | `det3d/models/readers/voxel_encoder.py:17-24` | `VoxelFeatureExtractorV3.forward()` | `features`, `num_voxels` | `points_mean` | voxel 내부 point feature 평균을 구한다. |
| 6 | `det3d/models/detectors/voxelnet.py:46-48` | `self.backbone(input_features, coors, batch_size, input_shape)` | voxel feature, coordinate, batch size, shape | `x`, `voxel_feature` | sparse backbone 호출. |
| 7 | `det3d/models/backbones/scn.py:162-191` | `SpMiddleResNetFHD.forward()` | sparse tensor 구성 요소 | BEV tensor, multi-scale voxel features | sparse conv 후 dense BEV feature로 변환한다. |
| 8 | `det3d/models/detectors/voxelnet.py:50-51` | `if self.with_neck: x = self.neck(x)` | backbone BEV feature | neck output feature | `BaseDetector.with_neck`은 `det3d/models/detectors/base.py:24-26`에 정의되어 있다. |
| 9 | `det3d/models/necks/rpn.py:150-159` | `RPN.forward()` | BEV feature | concat/upsample된 BEV feature | bbox head 입력 feature를 만든다. |
| 10 | `det3d/models/detectors/voxelnet.py:57` | `preds, _ = self.bbox_head(x)` | neck output | prediction dict list, final feature | `CenterHead.forward()` 호출. |
| 11 | `det3d/models/bbox_heads/center_head.py:235-243` | `CenterHead.forward()` | BEV feature | `ret_dicts, x` | task별 heatmap/regression head 출력을 만든다. |
| 12 | `det3d/models/detectors/voxelnet.py:61-62` | `self.bbox_head.predict(example, preds, self.test_cfg)` | example, preds, test_cfg | detection result list | return_loss=False이므로 predict branch로 간다. |
| 13 | `det3d/models/bbox_heads/center_head.py:292-447` | `CenterHead.predict()` | raw head outputs | sample별 detection dict list | decode, score threshold, NMS 후 metadata를 붙여 반환한다. |
| 14 | `det3d/models/bbox_heads/center_head.py:449-494` | `post_processing()` | decoded boxes, heatmap, test_cfg | `prediction_dicts` | score/distance mask와 rotate/circle NMS를 적용한다. |

`CenterHead.predict()`가 최종 반환하는 각 sample dict의 핵심 key:

| 파일:라인 | key | 의미 |
|---|---|---|
| `det3d/models/bbox_heads/center_head.py:486-490` | `box3d_lidar` | 선택된 3D box |
| `det3d/models/bbox_heads/center_head.py:486-490` | `scores` | box confidence |
| `det3d/models/bbox_heads/center_head.py:486-490` | `label_preds` | class label |
| `det3d/models/bbox_heads/center_head.py:444` | `metadata` | sample metadata. 이후 `token` 추출에 사용된다. |

## 8. prediction 저장 및 evaluation 흐름

| 단계 | 파일:라인 | 코드/함수 | 입력 | 출력 | 설명 |
|---:|---|---|---|---|---|
| 1 | `tools/dist_test.py:170` | `for output in outputs:` | `VoxelNet.forward()` 반환 list | sample별 output | batch 안의 sample 결과를 순회한다. |
| 2 | `tools/dist_test.py:171` | `token = output["metadata"]["token"]` | output metadata | sample token | prediction dict의 key로 쓸 token을 뽑는다. |
| 3 | `tools/dist_test.py:172-176` | `output[k] = v.to(cpu_device)` | tensor output | CPU tensor | metadata 외 tensor들을 CPU로 옮긴다. |
| 4 | `tools/dist_test.py:177-179` | `detections.update({token: output})` | token, output | `detections` dict | token 기준으로 detection 결과를 저장한다. |
| 5 | `tools/dist_test.py:183` | `synchronize()` | distributed workers | 동기화 | 모든 worker가 추론을 마칠 때까지 맞춘다. |
| 6 | `tools/dist_test.py:185` | `all_predictions = all_gather(detections)` | rank별 detections | list of detections | 분산 결과를 rank 0으로 모을 준비를 한다. |
| 7 | `tools/dist_test.py:189-190` | `if args.local_rank != 0: return` | local_rank | rank 0만 계속 진행 | 저장/평가는 rank 0만 수행한다. |
| 8 | `tools/dist_test.py:192-194` | `predictions.update(p)` | gathered predictions | 통합 predictions | rank별 dict를 하나로 합친다. |
| 9 | `tools/dist_test.py:196-199` | `os.makedirs(...)`, `save_pred(...)` | predictions, work_dir | `prediction.pkl` | `save_pred()`는 `tools/dist_test.py:32-34`에서 pickle로 저장한다. |
| 10 | `tools/dist_test.py:201` | `dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)` | predictions | `result_dict, _` | dataset별 평가 코드로 넘어간다. |
| 11 | `tools/dist_test.py:203-205` | `print(f"Evaluation {k}: {v}")` | `result_dict["results"]` | metric 출력 | 평가 결과가 있으면 콘솔에 출력한다. |

Dataset evaluation 연결:

| dataset type | 등록/생성 | evaluation |
|---|---|---|
| `NuScenesDataset` | `det3d/datasets/nuscenes/nuscenes.py:29-30` | `det3d/datasets/nuscenes/nuscenes.py:192` |
| `WaymoDataset` | `det3d/datasets/waymo/waymo.py:18-19` | `det3d/datasets/waymo/waymo.py:94` |

대표 NuScenes config에서는 `dataset_type = "NuScenesDataset"`가 `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:86`에 있고, val/test dict의 `type=dataset_type`는 `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py:182-193`에 있다.

## 9. 한눈에 보는 전체 데이터 흐름

```text
config file
→ Config.fromfile()
→ cfg.model
→ build_detector()
→ DETECTORS registry
→ VoxelNet
→ SingleStageDetector.__init__()
→ build_reader() / build_backbone() / build_neck() / build_head()
→ cfg.data.val 또는 cfg.data.test
→ build_dataset()
→ build_dataloader()
→ data_loader
→ batch_processor()
→ example_to_device()
→ model(example, return_loss=False)
→ VoxelNet.forward()
→ VoxelNet.extract_feat()
→ reader
→ backbone
→ neck
→ bbox_head.forward()
→ bbox_head.predict()
→ outputs
→ detections[token]
→ all_gather()
→ prediction.pkl
→ dataset.evaluation()
```

## 10. 주의할 점 / 헷갈리기 쉬운 부분

| 포인트 | 설명 |
|---|---|
| `build_detector()`가 `VoxelNet`을 직접 호출하지 않는다 | `tools/dist_test.py:106`은 `build_detector()`만 호출한다. 실제 class 선택은 `det3d/utils/registry.py:61-63`에서 `cfg["type"]` 문자열로 registry를 검색하면서 일어난다. |
| registry 등록은 import 시점의 decorator 부작용이다 | `det3d/models/detectors/voxelnet.py:7-8`의 `@DETECTORS.register_module`이 class를 등록한다. 이 파일은 `det3d/models/detectors/__init__.py:4`와 `det3d/models/__init__.py:17` 경로로 import된다. |
| `VoxelNet`은 `SingleStageDetector`를 상속한다 | `det3d/models/detectors/voxelnet.py:8`에서 상속하고, `det3d/models/detectors/voxelnet.py:19-21`에서 부모 초기화로 submodule 생성을 위임한다. |
| 추론에서는 loss가 아니라 predict가 호출된다 | `det3d/torchie/apis/train.py:113`에서 `return_loss=False`로 model을 호출하고, `det3d/models/detectors/voxelnet.py:61-62`에서 `bbox_head.predict()`를 반환한다. |
| `cfg.model.reader/backbone/neck/bbox_head`의 type에 따라 실제 class가 바뀐다 | 대표 config는 `VoxelFeatureExtractorV3`, `SpMiddleResNetFHD`, `RPN`, `CenterHead`지만, 다른 config에서 type이 바뀌면 같은 builder/registry 경로로 다른 class가 생성된다. |
| `DistributedDataParallel`이면 model 호출 앞에 wrapper가 낀다 | `tools/dist_test.py:126-134`에서 distributed일 때 `DistributedDataParallel(model.cuda(...))`로 감싼다. 그래도 내부 module의 forward 흐름은 `VoxelNet.forward()`로 이어진다. |
| `with_neck`은 boolean field가 아니라 property다 | `det3d/models/detectors/base.py:24-26`에서 `self.neck` 존재 여부로 판단한다. |
| `example_to_device()`는 key별 이동 방식이 다르다 | `det3d/torchie/apis/train.py:31-57`에서 `points`는 list 내부 tensor를 옮기고, `voxels/coordinates/num_points`는 tensor 자체를 옮긴다. |
| dataset도 model과 같은 registry 패턴이다 | `det3d/datasets/builder.py:31-42`가 `DATASETS` registry로 dataset type 문자열을 class로 바꾼다. |

