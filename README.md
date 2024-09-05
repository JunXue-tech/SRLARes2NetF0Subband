# SR-LA-Res2Net_F0-Subband: Spatial reconstructed local attention Res2Net with F0 subband for fake speech detection

This repository contains the code for a deep learning-based **ASV (Automatic Speaker Verification) spoof detection** system using **SR_LA_Res2Net** architecture. The model incorporates advanced techniques such as **A-Softmax Loss** and **Feature Subband Extraction (F0_subband)** to enhance the performance in detecting spoofing attacks on ASV systems. The project also includes learning rate scheduling, model evaluation, and data handling utilities to facilitate robust training and testing.

## Features

- **Res2Net-based architecture** with Local Attention (LA) and Spatial Reconstruction (SR) blocks.
- **A-Softmax loss function** for better margin learning in classification tasks.
- **F0 Subband Feature Extraction** as a core method to preprocess input audio data.
- **Learning Rate Scheduling** to improve training stability.
- **Pre-configured protocols** for training, development, and evaluation datasets (ASVspoof2019 dataset).
- **PyTorch** implementation for easy integration and customization.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Usage](#usage)
4. [Citing](#Citing)

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/SR_LA_F0_subband.git
    cd SR_LA_F0_subband
    ```

2. **Install required dependencies**:
   
   The dependencies for this project are listed in the `requirements.txt` file. To install them, run:

    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure CUDA is available** (optional):

   If you are using a GPU, ensure that PyTorch is installed with CUDA support. You can verify the availability of CUDA using:

    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

## Dataset Preparation

This project is designed to work with the **ASVspoof2019** dataset. Follow these steps to prepare the dataset:

1. Download the **ASVspoof2019 dataset** from the official challenge website: [ASVspoof2019](https://www.asvspoof.org/).
2. Extract the dataset and ensure the following structure for the protocols and audio files:

    ```
    ├── ASVspoof2019_LA_cm_protocols
    │   ├── ASVspoof2019.LA.cm.train.trl.txt
    │   ├── ASVspoof2019.LA.cm.dev.trl.txt
    │   └── ASVspoof2019.LA.cm.eval_test.trl.txt
    ├── ASVspoof2019_LA_train
    ├── ASVspoof2019_LA_dev
    └── ASVspoof2019_LA_eval
    ```

3. Update the file paths in your protocol files or scripts if necessary.

## Usage

### Configuration

Most configurations such as the number of epochs, learning rate, and batch size are defined as arguments. You can modify them by passing parameters when running the training script.

```bash
python main.py --batch-size 64 --epochs 32 --lr 0.0001 --gpu 0 --out_fold ./models/




## Citing:

Please cite our paper(s) if you find this repository useful.

``` 
@article{
fan2024spatial,
  title={Spatial reconstructed local attention Res2Net with F0 subband for fake speech detection},
  author={Fan, Cunhang and Xue, Jun and Tao, Jianhua and Yi, Jiangyan and Wang, Chenglong and Zheng, Chengshi and Lv, Zhao},
  journal={Neural Networks},
  pages={106320},
  year={2024},
  publisher={Elsevier}
}
```  

 ## Contact
 
If you have a question, please bring up an issue (preferred) or send me an email junxue.tech@gmail.com.
