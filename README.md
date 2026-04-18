# steel-fault-detection-u-net

My attempt at an old kaggle competition to get more perspective into encoding and multi-label detection! Firstly shoutout to these people and the resourses they created which taught me a lot:

1. https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
2. https://www.kaggle.com/code/amanooo/defect-detection-starter-u-net - best EDA on the dataset if you want to get started on learning about the problem
3. https://www.kaggle.com/code/liyufeng816/notebook-lyf
4. https://www.sciencedirect.com/science/article/pii/S259012302500060X

I have tried to combine these archetictures and give my own twist on it by:
1. making it light enough to train on an M1 (over 8 hours)
2. Image pre-processing to improve performance (tried couple of suggestions from forums, settled with light and noise distortion during training for generalisational ability)
3. tested out a bunch of loss function apporaches in unet5_8 (I learned my val scores were was class dependent and vanilla Dice Loss is prone to imbalance collapse) and tested two apporaches:
  a. an averaging approach to the calculationg global dice loss to combat it - MDL
  b. generalising based on weight of classes - GDL
5. a split dice coef calculation and tracking for better visibility on training

There were couple of attempts at training and I have maintainted training records of each archeticture for clarity and transparency, in case I can save someone a bit of research and training time - cheers!

