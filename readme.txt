
1. follow CenterFace github to setup the environment

2. training setting is in src/lib/opts_pose.py

3. python main.py to run the training process

4. python test_wider_face.py to test image

5. python evaluation.py to evaluate the widerface dataset result

6. python evaluation_MAFA.py to evaluate the MAFA dataset result

7  data mix is use for training else if for evaluation

8. pretrain on wider face is the trained res18 centernet model for initial weight

9. model is in src/lib/models/network/msra_resnet.py

10. loss is in src/lib/trains/ctdet.py
