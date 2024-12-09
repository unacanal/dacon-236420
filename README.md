# DACON 이미지 색상화 및 손실 부분 복원 AI 경진대회 [1st Place]
1st Place Solution for DACON Image Colorization and Restoration AI Competition


```mermaid
flowchart TD
    subgraph Phase1[Phase 1: Mask Generation]
        A[Test Input Image] --> B["Segment Anything Model (SAM)"]
        B --> C[Generated Mask]
    end
    subgraph Phase2[Phase 2: Image Inpainting]
        C --> D["MAT Model"]
        A --> D
        D --> E[Inpainted Result]
    end
    subgraph Phase3[Phase 3: Colorization]
        subgraph Training[DDColor Training]
            F[Train Image] --> G[DDColor Model]
            G -- Parameters --> H[Trained Model Weights]
        end
        
        subgraph Inference[DDColor Inference]
            E --> I[DDColor Model]
            H --> I
            I --> J[Final Colorized Result]
        end
    end
    %% Universal color scheme for both light and dark modes
style Phase1 fill:#9e9e9e,stroke:#bdbdbd,stroke-width:2px,color:#ffffff
style Phase2 fill:#757575,stroke:#9e9e9e,stroke-width:2px,color:#ffffff
style Phase3 fill:#7986cb,stroke:#9fa8da,stroke-width:2px,color:#ffffff
    style Training fill:#4db6ac,stroke:#80cbc4,stroke-width:1px,color:#ffffff
    style Inference fill:#7e57c2,stroke:#9575cd,stroke-width:1px,color:#ffffff

```

## 1. SAM (Segment Anything Model)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install segment_anything opencv-python
```

### Download SAM model
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
### Inference
```bash
cd 1_SAM
python find_mask.py
```
Generated masks will be saved in the `data/test_mask` directory.

## 2. Inpainting with [MAT (Mask-Aware Transformer)](https://github.com/fenglinglwb/MAT)

- Input: Test Input Images + Masks
- Output: Inpainted Images
- Download pretrained model '*Places_512_FullData.pkl*' from [One Drive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA?e=7OwJyE) and put it in the `2_MAT/pretrained` directory.

### Install dependencies
```bash
pip install -r requirements.txt
```
### Inference
```bash
cd 2_MAT
make inpaint
```
Inpainted test images will be saved in `data/test_inpainted` directory.


## 3. Colorization with [DDColor (Dual Decoder Colorization)](https://github.com/piddnad/DDColor)

Download pretrained model for fine-tuning and finetuned model, and install dependencies
```bash
cd 3_DDColor
make download

python setup.py develop
```

### Train
```bash
cd 3_DDColor
make train
# or make train_ddp
```
Trained models will be saved in `experiments/train_ddcolor_l/models` directory.

### Inference
```bash
cd 3_DDColor
make test
```
Colorized test images will be saved in `results/test_dacon/visualization/Dacon` directory.