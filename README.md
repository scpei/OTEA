# Improving Cross-lingual Entity Alignment via Optimal Transport

The study in this paper is to address two limitations that widely exist in current solutions: 
1) the alignment loss functions defined at the entity level serve well the purpose of aligning labeled entities but fail to match the whole picture of labeled and unlabeled entities in different KGs; 
2) the translation from one domain to the other has been considered (e.g., X to Y by M1 or Y to X by M2).

The implementation is based on the code and data of MTransE.

Contact: Shichao Pei (shichao.pei@kaust.edu.sa)

## Usage:

To run the code, you need to have Python3 and Tensorflow installed.

run `run_train_test.sh`

Visit https://drive.google.com/file/d/1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z/view to download the datasets.

## Dependencies
* Python>=3.5
* Tensorflow>=1.1.0
* numpy
* scipy
* multiprocessing
* pickle
* heapq

## Reference
Please refer to our paper. 

    @inproceedings{pei2019improving,
      title={Improving cross-lingual entity alignment via optimal transport},
      author={Pei, Shichao and Yu, Lu and Zhang, Xiangliang},
      booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence},
      pages={3231--3237},
      year={2019},
      organization={International Joint Conferences on Artificial Intelligence Organization}
    }
