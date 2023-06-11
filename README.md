
<h1 align="center">A Semantic Transferred Priori for Hyperspectral Target Detection with Spatial-Spectral Association [IEEE TGRS 2023]</h1>

<p align="center">

<a href=""> <img  src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

<h4 align="center">This is the official repository of the paper ''A Semantic Transferred Priori for Hyperspectral Target Detection with Spatial-Spectral Association''</a>.</h4>


<h5 align="center"><em>Jie Lei, Simin Xu, Weiying Xie<sup>&#8727;</sup>, Jiaqing Zhang, Yunsong Li, and Qian Du</em></h5>

# Train.
During the training phase, process the dataset according to ./dataset and run:
~~~
python train.py
~~~

# Test.
During the testing phase, run:
~~~
python test.py
~~~

# Matlab Code.
Run the ./matlab_code_Make_d/maked.m to generate a precise customized target spectrum for subsequent spectral detection. The corrected spectrum has been experimentally verified to significantly enhance the detection accuracy across various methods.
# Citation.
  
If you find this project useful for your research, please use the following BibTeX entry.

    @article{lei2023semantic,
      title={A Semantic Transferred Priori for Hyperspectral Target Detection With Spatial--Spectral Association},
      author={Lei, Jie and Xu, Simin and Xie, Weiying and Zhang, Jiaqing and Li, Yunsong and Du, Qian},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      volume={61},
      pages={1--14},
      year={2023},
      publisher={IEEE}
    }
