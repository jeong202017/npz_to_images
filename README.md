# npz_to_images

이 저장소는 Kaggle용 Wafer Map NPZ 파일에서 개별 wafer 이미지를 추출하여 PNG/JPG로 저장하는 간단한 유틸리티 코드(`npz_to_images.py`)를 포함합니다.

## 목적

사용자는 원본 Kaggle 데이터셋(원문 링크 아래)을 다운로드한 후 이 스크립트를 사용해 `.npz` 내부의 wafer map 배열을 개별 이미지 파일로 변환할 수 있습니다. 이 README는 Kaggle의 `Code` 섹션에 올릴 목적으로 작성되었습니다.

데이터셋 출처:

https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets/code

저작권/라이선스 관련 안내
- 데이터셋 자체의 라이선스와 배포 조건은 Kaggle 페이지(위 링크)에 명시되어 있으므로, 데이터 사용 전 해당 페이지의 라이선스 및 이용 약관을 반드시 확인하세요.
- 이 저장소는 데이터셋의 원본 파일(.npz)을 포함하지 않으며, 대용량 파일은 GitHub에 직접 커밋하지 마시고 Git LFS 또는 외부 스토리지를 사용하시길 권장합니다.

## 주요 기능
- `.npz` 파일에서 자동으로 이미지(X 배열)와 라벨(Y 배열) 배열을 추정합니다.
- 2D wafer map을 PNG/JPG로 저장합니다.
- 라벨이 존재하면 단일 결함은 해당 라벨명 폴더에, 복수 라벨(혼합)은 `Mixed` 폴더에 저장합니다.
- `meta.json` 파일에 변환에 사용된 메타 정보를 저장합니다.

## 요구사항
- Python 3.7+
- numpy
- matplotlib

설치(권장: 가상환경 사용)

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -U pip
pip install numpy matplotlib
```

## 사용법 (예시)

원본 `.npz` 파일(예: `Wafer_Map_Datasets.npz`)을 로컬에 다운로드한 후, 다음 예시처럼 실행합니다.

```powershell
# 예: 출력 폴더를 wafer_images_out로 지정, 색상맵 gray, 포맷 png
python npz_to_images.py --npz Wafer_Map_Datasets.npz --out wafer_images_out --fmt png --cmap gray
```

옵션 요약
- `--npz`: 입력 .npz 파일 경로 (필수)
- `--out`: 출력 폴더 경로 (필수)
- `--fmt`: 저장 포맷 (`png`, `jpg`, `jpeg`), 기본 `png`
- `--cmap`: matplotlib colormap (기본 `gray` 추천)
- `--dpi`: 저장 DPI (기본 200)
- `--no_label_folders`: 라벨 폴더 없이 모두 같은 폴더에 저장
- `--class_names_json`: 클래스 이름 JSON 문자열 또는 파일 경로 (선택)

예시: 로컬 `master` 브랜치의 코드를 원격의 `main`으로 올리고 싶을 때(참고용)

```powershell
git add npz_to_images.py
git commit -m "Add npz_to_images.py"
# (데이터 파일은 .gitignore에 추가하고 커밋하지 마세요)
git push -u origin main
```

## 출력 형식
- 출력 폴더 구조 예시:

```
wafer_images_out/
  meta.json
  Normal/
    wafer_000000.png
  Mixed/
    wafer_000001.png
  ...
```

`meta.json`에는 사용된 `.npz` 경로, 선택된 colormap, dpi, class_names 등 변환에 사용된 메타 정보가 기록됩니다.

## 대용량 데이터 주의사항
- `.npz` 파일처럼 큰 파일(예: 수백 MB)은 Git에 직접 커밋하지 마세요. 이미 커밋된 경우에는 히스토리에서 제거해야 하며(예: `git-filter-repo`, BFG 등 사용), 협업 상황에서는 주의해야 합니다.
- 권장: 데이터는 별도 스토리지(예: Google Drive, Kaggle Dataset 다운로드, S3 등)에 두고 코드만 리포지토리에 보관하세요.

## 기여 및 문의
- 간단한 버그 리포트나 개선 제안은 Issues에 남겨 주세요.
- 이 코드는 데이터 전처리 도구이며, 실제 모델 학습/평가는 제공하지 않습니다.

---

작성자: 리포지토리 소유자 또는 코드 기여자
# npz_to_images
Mixed-type Wafer Defect Datasets 에서 이미지 추출을 위한 코드
