# Alerta
An AI-powered real-time drowsiness detection system that uses computer vision to monitor driver fatigue, track eye closure (PERCLOS), and prevent road accidents with instant alerts.

## Project base and experiments

Built on the open-source [Drowsiness detection [OPEN CV]](https://github.com/shsarv/Machine-Learning-Projects/tree/main/Drowsiness%20detection%20%5BOPEN%20CV%5D) project
(Haar cascade eye detection + Keras CNN eye-state classifier, real-time alarm in `drowsinessdetection.py`).

Experiments added on top of it — baseline, data augmentation, and a learning-rate change, evaluated on the
Driver Drowsiness Dataset (DDD) with a subject-wise split — are documented with real results in [EXPERIMENTS.md](EXPERIMENTS.md).
