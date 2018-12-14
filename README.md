# pruning
To build and run:

Ensure that you have TensorFlow installed on your machine. 

Run commands:  
`pip3 install -e .`  
`cd pruning`  

To train model from scratch and evaluate pruned networks, run:  
`python run.py run`  

To load pre-trained model and evaluate pruned networks, run:  
`python run.py run --load-trained`  

To see results and analysis, open `experiments.ipynb`.
