# CenterPoint C++ 구현 코드 가이드 모음

통합 개념 문서 `CENTERPOINT_CPP_STUDY_MANUAL.md`와 함께 다음 순서로 읽는다.

| 단계 | 구현 | 상세 문서 |
|---|---|---|
| 02 | CPU Voxelization | `02_project/IMPLEMENTATION_GUIDE.md` |
| 03 | Pillar Feature Decoration | `03_pillar_feature_project/IMPLEMENTATION_GUIDE.md` |
| 04 | Dummy PFN 구조 + 실제 checkpoint PFN | `04_pfn_project/IMPLEMENTATION_GUIDE.md`, `04_pfn_project/CHECKPOINT_PFN_GUIDE.md` |
| 05 | Scatter to BEV | `05_scatter_project/IMPLEMENTATION_GUIDE.md` |
| 06 | CUDA Conv-BN-ReLU | `06_rpn_project/IMPLEMENTATION_GUIDE.md` |
| 06 Full | 전체 RPN CUDA | `06_rpn_project/FULL_RPN_IMPLEMENTATION_GUIDE.md` |
| 07 | CenterHead CUDA | `07_center_head_project/IMPLEMENTATION_GUIDE.md` |
| 08 | Box Decode + Rotated NMS | `08_decode_project/IMPLEMENTATION_GUIDE.md` |
| 09 | Waymo Sensor Archive Input Bridge | `09_full_pipeline_project/IMPLEMENTATION_GUIDE.md` |

각 문서는 `입출력 -> main 호출 흐름 -> 핵심 반복문 -> memory offset -> 검증 -> 한계` 순서로 실제 코드를 설명한다.
