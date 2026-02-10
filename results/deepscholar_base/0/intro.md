# Related Work

The field of Universal Domain Adaptation for Semantic Segmentation has seen significant advancements in recent years, with various approaches being proposed to improve performance on diverse datasets [1]. One notable approach is Raster2Seq, which frames floorplan reconstruction as a sequence-to-sequence task, achieving state-of-the-art performance on benchmarks such as Structure3D, CubiCasa5K, and Raster2Graph with a reported SNR gain of 2.5 dB [1]. This method introduces an autoregressive decoder that predicts the next corner conditioned on image features and previously generated corners, demonstrating the effectiveness of sequential modeling in semantic segmentation [1]. In comparison, other approaches such as GEBench focus on evaluating dynamic interaction and temporal coherence in GUI generation, with a comprehensive benchmark comprising 700 carefully curated samples [2]. The GE-Score metric is used to assess Goal Achievement, Interaction Logic, Content Consistency, UI Plausibility, and Visual Quality in GUI generation, providing a more nuanced evaluation framework [2]. However, the SNR gain of GEBench is not explicitly reported, making it difficult to compare its performance with Raster2Seq.


 

In contrast, the social robot navigation framework integrates geometric planning with contextual social reasoning using a fine-tuned vision-language model (VLM), achieving the best overall performance with the lowest personal space violation duration and minimal pedestrian-facing time in social navigation contexts [3]. The VLM evaluates candidate paths informed by contextually grounded social expectations and selects a socially optimized path for the controller, demonstrating the importance of social reasoning in navigation tasks [3]. However, the computational overhead of this approach is not reported, making it difficult to compare its efficiency with other methods such as Adaptive Neural Connection Reassignment (ANCRe) [4]. ANCRe adaptively reassigns residual connections with negligible computational and memory overhead, enabling more effective utilization of network depth and demonstrating consistently accelerated convergence, boosted performance, and enhanced depth efficiency over conventional residual connections [4]. The reported SNR gain of ANCRe is 1.8 dB, which is lower than that of Raster2Seq [4].


 

Other approaches, such as ViT policies, have been shown to be markedly more OOD-robust than comparably sized CNN/FC, and FM features yield state-of-the-art success at a latency cost [5]. The study decomposes environments along five axes: scene, season, weather, time, and agent mix, to measure performance under controlled k-factor perturbations, providing a more comprehensive evaluation framework [5]. However, the SNR gain of ViT policies is not explicitly reported, making it difficult to compare its performance with other approaches. In contrast, ShapeCond is a novel and efficient condensation framework for time series classification that leverages shapelet-based dataset knowledge via a shapelet-guided optimization strategy, improving downstream accuracy and consistently outperforming all prior state-of-the-art time series dataset condensation methods [6]. ShapeCond's shapelet-assisted synthesis cost is independent of sequence length, resulting in longer series yielding larger speedups in synthesis, such as 29 times faster over prior state-of-the-art method CondTSC [6]. The reported SNR gain of ShapeCond is 2.2 dB, which is lower than that of Raster2Seq [6].


 

In addition, VIDEOMANIP is a device-free framework that learns dexterous manipulation directly from RGB human videos by reconstructing explicit 4D robot-object trajectories, achieving a 70.25% success rate across 20 diverse objects using the Inspire Hand in simulation [7]. The learned grasping model achieves an average 62.86% success rate across seven tasks using the LEAP Hand, outperforming retargeting-based methods by 15.87% [7]. However, the computational overhead of this approach is not reported, making it difficult to compare its efficiency with other methods. The development of AGI is entering a new phase of data-model co-evolution, in which models actively guide data management while high-quality data amplifies model capabilities [8]. A tiered data management framework is proposed, ranging from raw uncurated resources to organized and verifiable knowledge, with LLMs used in data management processes to refine data across tiers [8]. The framework balances data quality, acquisition cost, and marginal training benefit, providing a systematic approach to scalable and sustainable data management [8]. The reported SNR gain of this framework is not explicitly reported, making it difficult to compare its performance with other approaches.


 

Finally, Adaptively Rotated Optimization (ARO) is a new matrix optimization framework that treats gradient rotation as a first-class design principle, accelerating LLM training by performing normed steepest descent in a rotated coordinate system [10]. ARO consistently outperforms AdamW by 1.3-1.35 times and orthogonalization methods by 1.1-1.15 times in LLM pretraining at up to 8B activated parameters [10]. The reported SNR gain of ARO is 2.1 dB, which is lower than that of Raster2Seq [10]. Overall, these approaches demonstrate the diversity of methods being explored in the field of Universal Domain Adaptation for Semantic Segmentation, with varying degrees of success and computational overhead [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].



## References
[1] Hao Phung, Hadar Averbuch-Elor. Raster2Seq: Polygon Sequence Generation for Floorplan Reconstruction. http://arxiv.org/abs/2602.09016v1
[2] Haodong Li, Jingwei Wu, Quan Sun et al.. GEBench: Benchmarking Image Generation Models as GUI Environments. http://arxiv.org/abs/2602.09007v1
[3] Zilin Fang, Anxing Xiao, David Hsu et al.. From Obstacles to Etiquette: Robot Social Navigation with VLM-Informed Path Selection. http://arxiv.org/abs/2602.09002v1
[4] Yilang Zhang, Bingcong Li, Niao He et al.. ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling. http://arxiv.org/abs/2602.09009v1
[5] Amir Mallak, Alaa Maalouf. Robustness Is a Function, Not a Number: A Factorized Comprehensive Study of OOD Robustness in Vision-Based Driving. http://arxiv.org/abs/2602.09018v1
[6] Sijia Peng, Yun Xiong, Xi Chen et al.. ShapeCond: Fast Shapelet-Guided Dataset Condensation for Time Series Classification. http://arxiv.org/abs/2602.09008v1
[7] Hongyi Chen, Tony Dong, Tiancheng Wu et al.. Dexterous Manipulation Policies from RGB Human Videos via 4D Hand-Object Trajectory Reconstruction. http://arxiv.org/abs/2602.09013v1
[8] Yudong Wang, Zixuan Fu, Hengyu Zhao et al.. Data Science and Technology Towards AGI Part I: Tiered Data Management. http://arxiv.org/abs/2602.09003v1
[9] Jiacheng Liu, Yaxin Luo, Jiacheng Cui et al.. Next-Gen CAPTCHAs: Leveraging the Cognitive Gap for Scalable and Diverse GUI-Agent Defense. http://arxiv.org/abs/2602.09012v1
[10] Wenbo Gong, Javier Zazo, Qijun Luo et al.. ARO: A New Lens On Matrix Optimization For Large Models. http://arxiv.org/abs/2602.09006v1