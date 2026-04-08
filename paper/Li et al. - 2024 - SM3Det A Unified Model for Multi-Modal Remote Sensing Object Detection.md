## SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection 

## Yuxuan Li, Xiang Li†, Yunheng Li, Yicheng Zhang, Yimian Dai, 

## Qibin Hou, Ming-Ming Cheng, Jian Yang† 

 PCA Lab, VCIP, Computer Science, NKU yuxuan.li.17@ucl.ac.uk, {yunhengli, zhangyc}@mail.nankai.edu.cn, {xiang.li.implus, yimian.dai, houqb, cmm, csjyang}@nankai.edu.cn † Corresponding Authors 

 Abstract 

 With the rapid advancement of remote sensing technology, high-resolution multi-modal imagery is now more widely accessible. Conventional object detection models are trained on a single dataset, often restricted to a specific imaging modality and annotation format. However, such an approach overlooks the valuable shared knowledge across multi-modalities and limits the model’s applicability in more versatile scenarios. This paper introduces a new task called Multi-Modal Datasets and Multi-Task Object Detection (M2Det) for remote sensing, designed to accurately detect horizontal or oriented objects from any sensor modality. This task poses challenges due to 1) the trade-offs involved in managing multi-modal modelling and 2) the complexities of multi-task optimization. To address these, we establish a benchmark dataset and propose a unified model, SM3Det (Single Model for Multi-Modal datasets and Multi-Task object Detection). SM3Det leverages a grid-level sparse MoE backbone to enable joint knowledge learning while preserving distinct feature representations for different modalities. Furthermore, we propose a novel consistency and synchronization optimization mechanism, allowing it to effectively handle varying levels of learning difficulty across modalities and tasks. Extensive experiments demonstrate SM3Det’s effectiveness and generalizability, consistently outperforming the combination of specialized models on individual datasets. 

 Code — github.com/zcablii/SM3Det Datasets — http://www.kaggle.com/datasets/greatbird/soi-det Extended Version — https://arxiv.org/pdf/2412.20665 

## Introduction 

 Remote sensing object detection (Yuan et al. 2025; Ni et al. 2025; Li et al. 2025, 2024a; Dai et al. 2024) typically involves multiple sensors employing different imaging mechanisms, resulting in diverse data modalities. Traditionally, detection models are developed for specific datasets associated with a single modality and a predefined format detection task (Li et al. 2024b; Yang et al. 2021; Dai et al. 2021), as shown in Figure 1 (b). This conventional approach overlooks the valuable and inherent joint knowledge 

 Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. 

 Spatially Aligned Dataset A 

 Dataset B 

 Dataset A 

 Dataset B 

 Extractor Extractor 

 Task Head 

 Extractor Extractor 

 Task 1 Head 

 Task 2 Head 

 Extractor 

 Task 2 Head 

 Task 1 Head 

 Figure 1: Comparison of tasks: (a) Spatially Aligned MultiModality, (b) Traditional Single Dataset, and (c) M2Det. M2Det aims to utilize a unified model for detecting objects in any modality, handling various detection tasks. 

 within a unified remote sensing context. Furthermore, airborne platforms such as UAVs and satellites often carry multiple sensors, making it critical to process images from various modalities simultaneously. Previous multi-source object detection methods (Liu, Chen, and Wang 2021; Zhang, Huang, and Kuruoglu 2024; Zhang et al. 2024) have heavily relied on scarce, impractical, and inflexible spatially wellaligned paired images and spatial alignment algorithms (Devaraj and Shah 2013; Ahamed et al. 2012). These methods are also limited to performing single-format detection tasks, as depicted in Figure 1 (a). Thus, it is essential to develop a unified model capable of handling all modalities without requiring spatially aligned image pairs and performing multiple format detection tasks (referred to as “multi-tasks” throughout the paper), which is not thoroughly studied. To the best of our knowledge and industrial experience, this task has the potential to serve as a foundational technology for emerging low-altitude economies and applications involving flying cars, drones and satellites. To fill this research gap, we propose a new task called Multi-Modal Datasets and Multi-Task Object Detection (M2Det). M2Det aims to detect objects in any given image, regardless of its modality, and across predefined detection tasks—whether horizontal bounding boxes or oriented bounding boxes—as illustrated in Figure 1 (c). The M2Det task is closely related to two key research areas: multi-dataset object detection (Wang et al. 2019; Zhou, Koltun, and Kr¨ahenb¨uhl 2022) and multi-task learning (Zhang and Yang 2021; Chen et al. 2018). However, 

# arXiv:2412.20665v2 [cs.CV] 10 Nov 2025 


the M2Det task presents unique challenges. In traditional multi-dataset object detection, even though images may have different attributes—such as natural images and paintings—they often share similar underlying concepts (optical concepts). A simple joint training approach is effective, with a single model trained on the combined dataset typically outperforming models trained on individual datasets (Wang et al. 2019). In contrast, multi-modal datasets in remote sensing—such as RGB (Xia et al. 2018; Sun et al. 2022a), SAR (Li et al. 2024c; Zhang et al. 2021), IR (Sun et al. 2022b), and multi-spectral images (for Photogrammetry and ISPRS)—exhibit fundamentally different pattern concepts (as in Figure 6). While certain common knowledge may be shared across these modalities, the significant differences in data representation create a substantial modality gap, complicating the integration of information across modalities. Additionally, remote sensing datasets often include diverse annotation types, such as horizontal (Li et al. 2020, 2024c) and oriented (Xia et al. 2018; Sun et al. 2022b) bounding boxes, further adding complexity to model learning. These challenges may impede traditional model learning and optimization in the following ways: 1) Representation Constraints: A dense model that shares the same parameters across multiple tasks and modalities may encounter limitations in representation capacity, as a single set of parameters may struggle to effectively fit the diverse distributions inherent in each dataset. 2) Optimization Inconsistencies: The varying learning difficulties across different modalities and tasks can lead to unsynchronized optimization rates or optimization directions for various components of the model. This inconsistency can result in conflicting optimization outcomes, adversely affecting the model’s ability to achieve different loss objectives. To address these challenges, we first establish a comprehensive benchmark dataset by merging SARDet-100K (Li et al. 2024c), DOTA (Xia et al. 2018), and DroneVehicle (Sun et al. 2022b), which collectively span SAR, optical, and infrared modalities. Subsequently, we propose a unified model, SM3Det, tailored for the M2Det task in remote sensing, addressing the challenges from both model architecture and model optimization perspectives: Model Architecture: We propose integrating a plug-andplay grid-level sparse Mixture of Experts (MoE) architecture into backbone networks, enabling the model to capture both shared knowledge and modality-specific representations. In contrast to prior multi-dataset object detection models that use hard-coded, image-level routing (Wang et al. 2019; Jain et al. 2024), our approach introduces grid-level experts with dynamic routing. These experts operate on spatial grid features, allowing the model to adaptively process information at a grid level, which is crucial for object detection tasks. Model Optimization: We introduce a novel dynamic submodule optimization (DSO) mechanism for model optimization consistency and synchronization. It adaptively adjusts the learning rates of various network components based on tailored policies. DSO accommodates the varying learning complexities across different tasks and modalities by balancing the relative convergence rate and guaranteeing optimization direction consistency. Unlike traditional techniques that 

 primarily modify loss weights or gradients—often lacking precise manipulation over specific network submodules or suffering from inefficiencies—our DSO provides finegrained control while maintaining optimization efficiency. Intensive experiments indicate that our unified single SM3Det model significantly outperforms individual models across all modality datasets. Our lightweight SM3Det variant not only demonstrates excellent performance but also features a substantially reduced number of parameters. Furthermore, the SM3Det model exhibits strong generalizability, enabling it to adapt to various backbones and detectors. Our contributions are summarized as follows: 

- We introduce a new task: Multi-Modal Datasets and     Multi-Task object detection in remote sensing using a uni-     fied detection model. 

- We propose the SM3Det model, which addresses the chal-     lenges of the M2Det task by offering innovative solutions     from both model architecture and model optimization per-     spectives. 

- Extensive experiments and analyses on the established     benchmark dataset demonstrate that our proposed sin-     gle model is effective and outperforms individual models     across all modalities. 

## Related Work 

### Multi-Dataset Object Detection 

 Multi-dataset object detection aims to leverage a diverse collection of datasets to learn general knowledge and achieve universal object detection. Leveraging multiple datasets in training has proven to be a highly effective strategy for enhancing the performance of deep learning models across various applications (Kapidis, Poppe, and Veltkamp 2021; Zhao et al. 2020; Yan et al. 2020; Zhang et al. 2025; Yang et al. 2019). This approach has also been widely explored in the domain of object detection. The DA network (Wang et al. 2019), for instance, employs specialized SE layers (Hu, Shen, and Sun 2018) that serve as domain-specific attention mechanisms for individual datasets. Universal-RCNN (Xu et al. 2020) introduces a partitioned detector trained across multiple datasets, integrating features through an inter-dataset graph-based attention module. Unidet (Zhou, Koltun, and Kr¨ahenb¨uhl 2022) advances this concept by proposing a unified label space and underscoring the importance of batch sampling strategies. Models trained on combined optical-concept datasets typically outperform those trained on individual datasets, as multi-dataset training can serve as a powerful form of data augmentation. However, the diverse imaging modalities in remote sensing present unique challenges for joint training. This area remains largely unexplored. 

### Multi-Task Learning 

 Multi-task learning involves utilizing a single model to learn multiple objectives, typically with multiple task heads and loss functions. In multi-task learning, various strategies (Chen et al. 2018; Sener and Koltun 2018; Guo et al. 2018; Kendall, Gal, and Cipolla 2018) have been developed 


to address task imbalances and optimize learning outcomes. GradNorm (Chen et al. 2018) focuses on correcting gradient imbalances during backpropagation by adjusting the gradient sizes for each task’s loss function. Methods like Multi-Gradient Descent Algorithm (Sener and Koltun 2018) employ Pareto optimization for gradient backpropagation, though they can be inefficient due to the additional gradient calculations required. Similar to GradNorm, DWA (Wang et al. 2019) also uses task losses to assess convergence rates, however, it dynamically adjusts the weight of each task’s loss instead. Uncertainty (Kendall, Gal, and Cipolla 2018) loss takes a different approach by incorporating homoscedastic uncertainty into the weighted loss function. Unlike loss reweighting or gradient manipulation, our method dynamically adjusts the learning rate for network submodules, enhancing multi-modal datasets and multi-task learning by maintaining optimization consistency. 

### Mixture of Experts (MoE) 

MoE (Jacobs et al. 1991; Jacobs and Jordan 1993) leverages multiple expert networks to provide rich features. Sparse MoE(Shazeer et al. 2017) further introduces sparsity, allowing for the scaling up of model size without dramatically increasing computational complexity. In multi-task learning, sparse MoE enables different expert networks to learn distinct discriminative features. Most sparse MoEbased multi-task methods (Chen et al. 2023; Yang et al. 2024) are grounded in transformer architectures, integrating experts into vision transformer backbone blocks to selectively activate different paths during inference. DeepMoE (Liu et al. 2020) borrow the concept of sparse MoE into CNN networks by treating the channels within each convolutional layer as experts, enhancing representational power by adaptively sparsifying and recalibrating channel features. In multi-dataset learning, recent work (Jain et al. 2024) employs MoE within vision transformers to route image-level features to specialized experts. However, sparse MoE for multi-modal datasets learning remains largely unexplored. Unlike prior methods that implement hard-coded, image-level routing (Wang et al. 2019; Jain et al. 2024), we propose to leverage MoE into backbone networks at the feature grid level. This enables experts to effectively process spatial features, learning both shared representations and distinct patterns across modalities. 

## Methods 

### Task Definition 

The proposed M2Det task is designed to utilize a unified model for detecting objects in images from any modality, handling various predefined detection tasks, such as horizontal and rotated bounding boxes. The significance of this task is evident in various real-world applications, including low-altitude economy (Jiang et al. 2023; Huang et al. 2024), aerial surveillance (Avola et al. 2021; Bozcan and Kayacan 2020), earth observation (Li et al. 2017; Anderson et al. 2017), and other research domains (Khan, Yanmaz, and Rinner 2014; Jensen 2016). For instance, plantforms equipped with M2Det models can fully leverage avail

 able multi-modal data while benefiting from simplified version control and the seamless integration of multiple sensors without requiring model updates on the device. This significantly reduces model maintenance costs in industrial applications. Furthermore, processing images of different modalities in a single model within one mini-batch maximizes the parallel computing capabilities of GPUs, thereby enhancing computational and energy efficiency on edge devices. 

### Methodological Overview 

 The overall network architecture follows the classic design of multi-task learning models (Wang et al. 2019; Zhou, Koltun, and Kr¨ahenb¨uhl 2022). It consists of a relatively heavy feature space shared component (backbone) and relatively lightweight feature space independent components (task heads). The backbone is responsible for joint representation learning, with most parameters being shared, thus ensuring parameter efficiency. The lightweight heads are separated to accommodate distinct features and task learning. However, as discussed in Section , modality and task gaps may degrade the performance of such classic multitask models. To address this issue, we propose the SM3Det model, which consists of two parts: Model architecture: A sparse MoE backbone where experts are activated on local image features of multi-modality dataset images at the grid level. Model optimization: An efficient dynamic submodule optimization mechanism, to handle the varying learning difficulties and optimization inconsistency across multiple tasks and modalities. 

### Grid-level MoE 

 Previous approaches to multi-dataset object detection (Zhou, Koltun, and Kr¨ahenb¨uhl 2022; Xu et al. 2020) utilize dense models that leverage shared concepts among datasets to enhance joint knowledge representation. In the case of multimodal remote sensing images, this joint knowledge also exists (Li et al. 2024c), though it may be less explicit, with common weak cues such as shape and scale across modalities. However, due to inherent modality and task gaps, employing a dense model that utilizes the same parameters across multiple tasks and modalities can result in a congested feature/representation space, ultimately reducing the model’s expressiveness. Therefore, it is essential to explore methods that leverage joint knowledge across modalities while enabling distinct representation learning for each modality to prevent feature space interference. Drawing inspiration from the success of Sparse MoE networks (Shazeer et al. 2017), which are characterized by their sparsity and high capacity, we propose leveraging MoE for the M2Det task. For transformer-based backbones (Liu et al. 2021; Wang et al. 2022), we integrate MoE experts within the FFN components. For modern CNNs (Liu et al. 2022; Guo et al. 2022; Li et al. 2023b), which often employ 1× 1 convolutions (Lin 2013) for feature interaction or dimensionality reduction/expansion, we introduce sparse experts to enhance these layers. Unlike previous transformer-based detectors that route an entire image’s features through a single expert (Jain et al. 2024), our design allows experts to 


 … 

 Expert 1 

 Expert 2 

 Expert N 

 … 

 … 

 … 

 … 

 Multi-modal Datasets 

 Multi-task Heads 

 Parameter Shared Backbone 

 … 

 … 

 … 

 Grid-Level MoE 

 … Block 

 FFN / 

 1 

 ×^ 

 1 

 Conv 

 Block Block Grid 

**_- Level_** 

 MoE … 

 Dynamic Submodule Optimization (DSO) Task_1 Loss 

 Task_k Loss 

 Historical Losses (EMA update) optimizer OBB head 

 optimizer HBB head 

 Backbone LR Adjust (Eq. 9) 

 Head LR Adjust (Eq. 5) … 

 optimizer 

 Spatially Dispatch 

 Spatially Combine 

 Gating Network 

 Data Stream Parameter optimizer 

 Figure 2: A conceptual illustration of SM3Det model. “HBB”: horizontal bounding box, “OBB”: oriented bounding box. 

operate on local grid features within the backbone. This approach ensures that experts process similar spatial patterns across modalities, facilitating shared representation learning. Simultaneously, multiple experts capture distinct patterns across modalities, enabling independent representation learning. Specifically, for the local spatial input feature xij at the i-th row and j-th column of a deep image feature, the output feature fM oE (xij ) after the MoE layer is: 

 fM oE (xij ) = 

#### X^ N 

 n=1 

 Gn(xij ) · Conv^1 n× 1 (xij ), (1) 

 G(xij ) = TOPk 

####  

 Softmax 

####  

 ET^ W xij τ ∥W xij ∥∥E∥ 

####  

#### , (2) 

where N is the total number of experts, G is the gating function and Conv^1 n× 1 is the n-th 1×1 convolutional expert. Each expert has a representation embedding in the matrix E. The input feature x is first transformed by the matrix W. The product of W x is then compared with each expert embedding in E to calculate the similarity. This comparison is then normalized by the product of the norms of W x and E, ensuring scale-invariance. The similarity scores are passed through a Sof tmax function, converting them into a probability distribution. This means the gating function assigns a probability to each expert, indicating its relevance to the input feature x. Finally, the TOPk operator selects the topk experts with the highest probabilities. It reweights each expert by assigning the Sof tmax probability to the topk experts, setting the rest to zero. This step sparsifies the model by focusing only on a small subset of experts, reducing computational complexity and enhancing the model’s expressiveness to handle diverse tasks and modalities. In summary, fM oE (xij ) is a weighted sum of the outputs from top-k experts. The weights are determined by the gating function G, which dynamically selects the most relevant expert(s) for each local feature. The MoE creates a sparser feature space in the backbone model. By focusing on local 

 patterns, the model can learn independently to model multiple modalities and local object patterns. Our design effectively addresses the challenges of crowded feature spaces and enhances the expressiveness of the model. In practical implementation, to fully utilize the pretrained backbone weights, we initialize the weights of added experts by duplicating the corresponding pretrained 1 × 1 convolutional layers’ weights before downstream model fine-tuning, ensuring all experts can be evenly chosen at the beginning of fine-tuning. For the task heads, we maintain simplicity and adhere to the existing design of task heads as in (Wang et al. 2019; Zhou, Koltun, and Kr¨ahenb¨uhl 2022). 

### Dynamic Submodule Optimization (DSO) 

 In multi-modal, multi-dataset, and multi-task object detection tasks, one primary challenge is the varying learning difficulties (Kendall, Gal, and Cipolla 2018; Chen et al. 2018) across modalities and tasks. The variation can cause unsynchronized optimization rates and inconsistent optimization directions (Nakano, Chen, and Demachi 2021), leading to conflicting objectives among different loss functions. To address this problem, we propose a novel Dynamic Submodule Optimization (DSO) mechanism to manage the differing learning difficulties across tasks and modalities. DSO takes each task head’s loss as indicator to determine the current convergence rate of each task and the overall optimization direction of the network, adjusting the learning rate (LR) accordingly. Specifically, one policy is for the LR of each task head submodule (non-shared weights) to balance each task’s relative convergence rate, and another policy is for the backbone submodule (shared weights) to ensure optimization direction consistency. We denote the training loss from the iteration i of task t as cur Lti. Each task’s loss maintains an exponential moving average (EMA) value as the smoothed historical statistic, denoted as his Lti, i.e., his Lti = α · cur Lti + (1 − α) · his Lti− 1. (3) 


 1 

 b 

 2 

 0 1 

 C 

#### 𝜸 

 DroneVehicle DOTA-v1.0 SARDet-100K 

 Figure 3: Reweighting curves for various temperature (τ ). 

For the head submodule’s LR adjustment, we use the ratio of his L to cur L as the inverse of the convergence rate for iteration i of task t as: 

 wit = 

 his Lti cur Lti 

#### . (4) 

The Sof tmax with temperature θ is then used to reweight the LR of the corresponding network task head, aiming to balance the convergence speed of each task. The reweighting factor λti for task t at training iteration i is denoted as: 

 λti = 

 T · ew it /θ PT k e 

 wki /θ ,^ (5) 

where T is the total number of tasks. As a result, a relatively large value of cur Lti indicates faster convergence for task t, leading to a smaller wti and, consequently, a lower reweighting factor λti to prevent overly rapid convergence. Conversely, a smaller value of cur Lti results in a larger λti. This strategy ensures that the convergence rate of each task remains balanced throughout training. For the backbone submodule’s LR adjustment, the reweighting is based on the historical consistency of each loss. To measure the training convergence consistency, we define a consistency score C based on cur L and his L. Specifically, cur L and his L are first converted into probability distributions using the function P , which employs a simple Sof tmax function: 

 P (L) = Sof tmax(L). (6) 

Next, the Kullback-Leibler divergence, DKL, is calculated to evaluate whether the current losses from each task remain stable and consistent with their historical values: 

 C = 1 − DKL ( P (cur L) ∥ P (his L) ) (7) 

 = 1 − 

#### XT 

 t P^ (cur L 

 t) · log P^ (cur L 

 t) 

 P (his Lt) 

#### , (8) 

therefore C is in the range of (−∞, 1]. A larger C indicates that the relative values of the current iteration losses are similar to their historical values, suggesting that the current batch of samples stabilizes the network updates. In this 

 case, the LR has to be increased to make the network converge faster. Conversely, a lower C indicates instability, suggesting that the current samples make some tasks more difficult and others easier to learn compared to the previous average state. If the network updates the shared weights too aggressively in such cases, the network will be optimized in the direction of the harder task of the current iteration, which might harm the easier tasks. Therefore, the network should update cautiously to reduce the LR. To balance this, we propose dynamically reweighting the shared weight backbone with the following policy: 

 γi = 2 · Sigmoid((C − b) · τ ) (9) 

 = 

#### 2 

 1 + e−(C−b)·τ^ 

#### . (10) 

 The scalar factor of 2 ensures the reweighted value after the sigmoid function is in the range of (0, 2). b is the hyperparameter, bias, which can be interpreted as the reweithing threshold, i.e. when the C is b, the reweight is 1. τ is the temperature for value sensitivity adjustment. The reweighting curves for various temperatures and the relation between b and C are demonstrated in Figure 3. 

## Experiments and Analysis 

 To train and evaluate models for the M2Det task, we establish a new benchmark dataset by merging three detection datasets: SARDet-100K (Li et al. 2024c), DOTA-v1.0 (Xia et al. 2018), and DroneVehicle (Sun et al. 2022b), which correspond to SAR, optical, and infrared modalities, respectively. We refer to this combined dataset as the SOI-Det dataset. More detailed dataset description and implementation details can be found in the Appendix. In the main results and ablation studies, ConvNext-T is used as the default backbone unless otherwise specified. 

### Main Results 

 We evaluate the performance of our proposed SM3Det model against individual dataset training, simple joint training, and three SOTA methods that can be adapted for this task: UniDet (Zhou, Koltun, and Kr¨ahenb¨uhl 2022) with a partitioned head, the DA network (Wang et al. 2019) implemented within the ConvNext-T backbone, and uncertainty loss (Kendall, Gal, and Cipolla 2018) implemented upon UniDet. The main results are presented in Table 1. It can be observed that simple joint training of the three multi-modality datasets—i.e., merely merging the datasets and using a model with a shared backbone and separate task heads, along with a random data sampling strategy—results in a significant performance drop. This phenomenon highlights the increased challenge of this task compared to multi-dataset training for general object detection, where simple joint training typically enhances the performance of individual datasets (Wang et al. 2019; Zhou, Koltun, and Kr¨ahenb¨uhl 2022). The previous SOTA methods, UniDet (Zhou, Koltun, and Kr¨ahenb¨uhl 2022), DA (Wang et al. 2019) and uncertainty loss (Kendall, Gal, and Cipolla 2018), barely exceed the baseline by a small 


 Model FLOPs #P Test on mAP @50 @75 3 models 403G 126M Overall 48.23 79.39 51.26 GFL 131G 36M SARDet-100K 57.31 87.44 61.99 O-RCNN 136G 45M DOTA 45.31 77.70 46.45 O-RCNN 136G 45M DroneVehicle 46.09 74.78 52.79 

 Simple Joint Training 

#### 403G 66M 

 Overall 47.05 77.56 50.11 SARDet-100K 53.46 84.11 57.29 DOTA 45.18 76.37 46.78 DroneVehicle 44.99 73.28 51.50 

 DA +ConvNext-T 403G^ 66M 

 Overall 48.37 79.76 51.66 SARDet-100K 53.86 84.93 58.09 DOTA 46.23 78.47 47.58 DroneVehicle 48.21 77.43 56.16 

 UniDet (Partitioned) 

#### 403G 66M 

 Overall 48.47 79.55 52.01 SARDet-100K 53.81 84.70 57.43 DOTA 46.49 78.28 48.59 DroneVehicle 47.99 77.17 55.74 

 Uncertainty loss 

#### 403G 66M 

 Overall 48.79 79.99 52.50 SARDet-100K 53.43 84.81 57.41 DOTA 46.94 78.73 49.08 DroneVehicle 48.78 77.96 56.88 

 SM3Det (DSO only) 

#### 403G 66M 

 Overall 49.40 80.19 52.93 SARDet-100K 58.54 88.59 62.67 DOTA 46.18 77.86 47.95 DroneVehicle 48.09 77.09 56.20 

 SM3Det 487G 178M 

 Overall 50.20 80.68 53.79 SARDet-100K 60.64 89.94 65.06 DOTA 46.47 77.88 48.24 DroneVehicle 48.87 77.99 56.90 

Table 1: Model performance comparison on the SOI-Det dataset (SARDet-100K + DOTA + DroneVehicle). The proposed SM3Det model outperforms individual models and other SOTA models. 

margin. In contrast, our proposed SM3Det model significantly improves overall mAP performance from 48.23 to 50.20, an increase of 1.97 mAP. To be noticed, our lightweight version of SM3Det which only incorporates DSO but without MoE structures, also easily outperforms other SOTA methods. To assess the generalization capability of SM3Det, we evaluate its performance across different backbones and detectors. As illustrated in Figure 4, SM3Det significantly outperforms individual models across various modern convolutional backbones, including ConvNext (Liu et al. 2022), VAN (Guo et al. 2022), LSKNet (Li et al. 2023b) and PVTv2 (Wang et al. 2022). The model also exhibits reasonable scalability as the model size increases. We also evaluate SM3Det with different detectors. Since both the optical dataset (DOTA) and the infrared dataset (DroneVehicle) involve OBB regression tasks, we use the same head network structure in our model. In contrast, for the SAR dataset (SARDet-100K), which involves an HBB regression task, we implement a standard horizontal object detection head. Figure 5 shows our evaluation of SM3Det on onestage (RetinaNet (Lin et al. 2017), GFL (Li et al. 2022) and S^2 ANet (Han et al. 2020)) and two-stage (F-RCNN (Ren et al. 2015), Cascade F-RCNN (Cai and Vasconcelos 2018), O-RCNN (Xie et al. 2024) and RoI-Transformer (Ding 

 图表标题 

 ConvNext VAN LSKNet ConvNext2 

 Base 

 51 

 49 

 47 

 45 

 43 

 ConvNext LSKNet 

 VAN 

 SM3Det Individuals Tiny Small 

 PVT-v2 

 Figure 4: SM3Det on different backbones. 

 et al. 2019)) detector combinations. The results consistently demonstrate that SM3Det significantly outperforms individual models across all detector combinations. 

### Ablation Study and Analysis 

 Expert Number and top-k Number. In sparse MoE architecture, the number of experts to add (N ) and the top-k value play crucial roles in determining the model’s performance and efficiency. Increasing N generally enhances the model’s representation capacity, while a higher top-k value allows 


 MoE (N , k) w/o MoE 2, 2 4, 2 6, 2 8, 2 10, 2 8, 1 8, 2 8, 3 8, 2 Image-Level 

#### 8, 2 

 Grid-Level FLOPs (G) 403 469 469 469 469 469 403 469 531 487 487 #P (M) 66 82 113 142 174 205 174 174 174 178 178 mAP 48.51 48.94 49.11 49.13 49.31 49.24 49.05 49.31 49.13 48.25 50.20 @50 79.70 80.25 80.10 79.74 80.26 80.18 79.72 80.26 79.98 79.10 80.68 @75 51.78 52.01 52.13 52.76 52.84 52.79 52.30 52.84 52.77 51.31 53.79 

Table 2: Experiments on the MoE backbone with varying numbers of experts and top-k selection configurations. Experts are applied only to the even-indexed layers of the last two stages for validation efficiency, except for the last 2 columns. N : number of experts to add. k: number of experts to activate. The optimal configuration balancing performance and computational efficiency is identified as 8 experts with a top-k value of 2. 

 One-Stage 

 Two-Stage 

 One-Stage Two-Stage 

 OBB tasks F-RCNN + S^2 ANet 

 Cascade + S^2 ANet 

 F-RCNN + O-RCNN 

 Cascade + RoI-Trans 

 RetinaNet + RoI-Trans 

 GFL + O-RCNN 

 GFL + S^2 ANet 

 RetinaNet + S^2 ANet 

 HBB task 

 48.01 

 45.67 

 43.13 41.99 

 42.81^ 43.76 

 48.59 

 46.50 48.89 

 47.10 

 43.12 42.29 

 50.20 

 48.23 

 45.25 43.43 

 Individual Models SM3Det Model 

 Figure 5: SM3Det on different detector heads. 

for more specialized knowledge to be applied to each input. However, these enhancements come at the cost of a larger model size, increased computational complexity, and potentially requiring more training data to ensure that each expert is adequately trained. Therefore, selecting the appropriate number of experts and top-k value is critical for achieving an optimal balance between model performance and computational efficiency. The results in Table 2 underscore the importance of tuning the number of experts and the top-k value in a sparse MoE architecture. It reveals that the optimal configuration for this sparse MoE architecture in terms of balancing performance and computational efficiency is 8 experts with a top-2 experts. This configuration maximizes the model’s ability to learn from diverse inputs without introducing unnecessary complexity or overfitting. 

Image-level v.s. Grid-level MoE. In Table 2, the grid-level MoE outperforms the image-level counterpart, indicating that grid-level experts more effectively capture spatial variations across different objects in multi-modal images. By processing features at a finer spatial granularity, experts are more attuned to object localization, making grid-level MoE particularly well-suited for object detection tasks. 

Grid-level Experts Activation Behaviour Analysis. We visualize the selection results for each grid area across the last three stages of a well-tuned ConvNext-T backbone. In this visualization, each square grid represents the corresponding receptive field of that stage, with local deep features pro

 DOTA (RGB) 

 DroneVehicle (Infar-Red) 

 SARDet100K (SAR) 

 Input Image Stage 2 Experts Stage 3 Experts Stage 4 Experts Experts 1 ： 

 8 ： 

 2 ： 3 ： 4 ： 5 ： 6 ： 7 ： 

 Figure 6: Visualization of grid expert activation across the last three stages of a well-tuned backbone on SAR, RGB, and IF images. Each square grid represents the receptive field at a given stage, with different colors indicating the local grid areas processed by distinct experts. The top-1 selected experts for each grid are shown. Each expert specializes in processing unique local patterns and semantics. 

 cessed by different experts indicated by distinct colours. The top-1 selected experts are illustrated in Figure 6. For both RGB and IR images, a consistent pattern emerges: expert 1 predominantly processes salient objects, while Expert 3 focuses on background patches across all three stages. In contrast, the situation is more complex for SAR images. Particularly at stage 4, three experts (Expert 1, Expert 4, and Expert 6) are responsible for processing background areas, with Expert 1 also handling ship objects. Additional visualizations and detailed MoE behaviour analysis can be found in the Supplementary Material. DSO hyperparamers. We conduct an ablation study on each component of the proposed DSO method, as well as the sensitivity of its two key hyperparameters. The results are summarized in Table 3. Omitting the learning rate adjustment for either the head or backbone leads to significant performance degradation. The bias parameter b and temperature τ dynamically adjust learning rates to account for varying task and modality difficulties. Specifically, b serves as a reweighting balance point, meaning when the calculated consistency score equals b, the reweighting factor is 1. A bias value of b = 0. 4 proved optimal when the temperature was fixed at 3, striking a good balance in learning rate adjustments. Notably, variations in b did not significantly impact performance, indicating that the method is robust to changes 


 τ , b 3, 0.3 3, 0.4 3, 0.5 3, 0.6 2, 0.4 3, 0.4 4, 0.4 w/o DSO w/o Head policy w/o Backbone policy mAP 50.14 50.20 50.07 50.03 49.92 50.20 50.03 49.47 49.86 50.11 @50 80.61 80.68 80.66 80.61 80.55 80.68 80.44 80.33 80.53 80.66 @75 53.81 53.79 54.00 53.98 53.56 53.79 53.79 52.98 53.44 53.70 

 Table 3: Experiments on the DSO method with varying temperature (τ ) and bias (b). DSO is not sensitive to bias b. 

in bias. Regarding temperature,τ , influences the reweighting curve in both the network head and the backbone’s learning rate adjustment mechanism. Larger values result in sharper, more sensitive adjustments. A temperature of τ = 3 provided the best balance between stability and responsiveness. In summary, the τ = 3 and b = 0. 4 yielded the best performance, effectively managing learning rate adjustments across diverse tasks and datasets. 

## Limitation and Future Work 

An important modality in remote sensing is multi-spectrum imaging. However, due to the limited availability of largescale multi-spectrum object detection datasets, we could not include such datasets in our experiments. To the best of our knowledge, the proposed M2Det task holds significant potential as a foundational technology for emerging lowaltitude economies (e.g., flying cars and drones). Moreover, our model designs and observations can extend beyond remote sensing to other scenarios involving multiple modalities or the joint training of diverse datasets, including medical imaging (Elangovan and Jeyaseelan 2016) (X-Ray, NMR and CT) and autonomous driving (Feng et al. 2020) (camera, LiDAR and Radar). These applications present exciting avenues for future research. 

## Conclusion 

In conclusion, this paper introduces a new and challenging task of Multi-Modal Datasets and Multi-Task Object Detection in remote sensing. To tackle this, we developed the SM3Det model, integrating a novel grid-level MoE approach and a dynamic submodule optimization mechanism. Intensive experiments and thorough analysis demonstrate SM3Det’s strong performance and generalizability. 

## References 

 Ahamed, T.; Tian, L.; Jiang, Y.; Zhao, B.; Liu, H.; and Ting, K. C. 2012. Tower remote-sensing system for monitoring energy crops; image acquisition and geometric corrections. Biosystems engineering, 112(2): 93–107. Anderson, K.; Ryan, B.; Sonntag, W.; Kavvada, A.; and Friedl, L. 2017. Earth observation in service of the 2030 Agenda for Sustainable Development. Geo-spatial Information Science. Avola, D.; Cinque, L.; Di Mambro, A.; Diko, A.; Fagioli, A.; Foresti, G. L.; Marini, M. R.; Mecca, A.; and Pannone, D. 

2021. Low-altitude aerial video surveillance via one-class SVM anomaly detection from textural features in UAV im- ages. Information. Bozcan, I.; and Kayacan, E. 2020. Au-air: A multi-modal unmanned aerial vehicle dataset for low altitude traffic surveillance. In 2020 IEEE International Conference on Robotics and Automation. Cai, Z.; and Vasconcelos, N. 2018. Cascade R-CNN: Delv- ing Into High Quality Object Detection. In CVPR. Chen, T.; Chen, X.; Du, X.; Rashwan, A.; Yang, F.; Chen, H.; Wang, Z.; and Li, Y. 2023. Adamv-moe: Adaptive multi-task vision mixture-of-experts. In ICCV. Chen, Z.; Badrinarayanan, V.; Lee, C.-Y.; and Rabinovich, A. 2018. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In ICML. Dai, Y.; Wu, Y.; Zhou, F.; and Barnard, K. 2021. Attentional local contrast networks for infrared small target detection. TGRS. Dai, Y.; Zou, M.; Li, Y.; Li, X.; Ni, K.; and Yang, J. 2024. DenoDet: Attention as Deformable Multi-Subspace Feature Denoising for Target Detection in SAR Images. arXiv. Devaraj, C.; and Shah, C. A. 2013. Automated geometric correction of Landsat MSS L1G imagery. IEEE Geoscience and Remote Sensing Letters, 11(1): 347–351. Ding, J.; Xue, N.; Long, Y.; Xia, G.-S.; and Lu, Q. 2019. Learning RoI Transformer for Oriented Object Detection in Aerial Images. In CVPR. Elangovan, A.; and Jeyaseelan, T. 2016. Medical imaging modalities: a survey. In International Conference on emerg- ing trends in engineering, technology and science. Feng, D.; Haase-Sch¨utz, C.; Rosenbaum, L.; Hertlein, H.; Glaeser, C.; Timm, F.; Wiesbeck, W.; and Dietmayer, K. 

2020. Deep multi-modal object detection and semantic seg- mentation for autonomous driving: Datasets, methods, and challenges. IEEE Transactions on Intelligent Transporta- tion Systems. 


for Photogrammetry, T. I. S.; and (ISPRS), R. S. 2022. 2D Semantic Labeling Contest Potsdam. https: //www.isprs.org/education/benchmarks/UrbanSemLab/2dsem-label-potsdam.aspx. 

Guo, M.; Haque, A.; Huang, D.-A.; Yeung, S.; and Fei-Fei, L. 2018. Dynamic task prioritization for multitask learning. In ECCV. 

Guo, M.-H.; Lu, C.; Liu, Z.-N.; Cheng, M.-M.; and Hu, S. 

2022. Visual Attention Network. Computational Visual Me- dia. 

Han, J.; Ding, J.; Li, J.; and Xia, G.-S. 2020. Align Deep Features for Oriented Object Detection. TGRS. 

Hu, J.; Shen, L.; and Sun, G. 2018. Squeeze-and-excitation networks. In CVPR. 

Huang, C.; Fang, S.; Wu, H.; Wang, Y.; and Yang, Y. 2024. Low-Altitude Intelligent Transportation: system architecture, infrastructure, and key technologies. Journal of Industrial Information Integration, 100694. 

Jacobs, R. A.; and Jordan, M. I. 1993. Learning piecewise control strategies in a modular neural network architecture. IEEE Transactions on Systems, Man, and Cybernetics. 

Jacobs, R. A.; Jordan, M. I.; Nowlan, S. J.; and Hinton, G. E. 

1991. Adaptive mixtures of local experts. Neural computa- tion. 

Jain, Y.; Behl, H.; Kira, Z.; and Vineet, V. 2024. DAMEX: Dataset-aware Mixture-of-Experts for visual understanding of mixture-of-datasets. NeurIPS. 

Jensen, O. B. 2016. Drone city–power, design and aerial mobility in the age of “smart cities”. Geographica Helvetica. 

Jiang, Y.; Li, X.; Zhu, G.; Li, H.; Deng, J.; and Shi, Q. 

2023. 6G Non-Terrestrial networks enabled low-altitude economy: Opportunities and challenges. arXiv preprint arXiv:2311.09047. 

Kapidis, G.; Poppe, R.; and Veltkamp, R. C. 2021. Multidataset, multitask learning of egocentric vision tasks. TPAMI. 

Kendall, A.; Gal, Y.; and Cipolla, R. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR. 

Khan, A.; Yanmaz, E.; and Rinner, B. 2014. Information merging in multi-UAV cooperative search. In 2014 IEEE international conference on robotics and automation. 

Li, B.; Shen, Y.; Yang, J.; Wang, Y.; Ren, J.; Che, T.; Zhang, J.; and Liu, Z. 2023a. Sparse mixture-of-experts are domain generalizable learners. In ICLR. 

Li, D.; Wang, M.; Dong, Z.; Shen, X.; and Shi, L. 2017. Earth observation brain (EOB): An intelligent earth observation system. Geo-spatial information science. 

Li, K.; Wan, G.; Cheng, G.; Meng, L.; and Han, J. 2020. Object detection in optical remote sensing images: A survey and a new benchmark. ISPRS. 

Li, W.; Yang, W.; Liu, T.; Hou, Y.; Li, Y.; Liu, Z.; Liu, Y.; and Liu, L. 2024a. Predicting gradient is better: Exploring selfsupervised learning for SAR ATR with a joint-embedding predictive architecture. ISPRS Journal o. 

 Li, X.; Lv, C.; Wang, W.; Li, G.; Yang, L.; and Yang, J. 

2022. Generalized focal loss: Towards efficient represen- tation learning for dense object detection. TPAMI. Li, Y.; Hou, Q.; Zheng, Z.; Cheng, M.-M.; Yang, J.; and Li, X. 2023b. Large Selective Kernel Network for Remote Sens- ing Object Detection. In ICCV. Li, Y.; Li, X.; Dai, Y.; Hou, Q.; Liu, L.; Liu, Y.; Cheng, M.-M.; and Yang, J. 2024b. LSKNet: A Foundation Lightweight Backbone for Remote Sensing. arXiv preprint arXiv:2403.11735. Li, Y.; Li, X.; Li, W.; Hou, Q.; Liu, L.; Cheng, M.-M.; and Yang, J. 2024c. SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detec- tion. In NeurIPS. Li, Y.; Zhang, Y.; Tang, W.; Dai, Y.; Cheng, M.-M.; Li, X.; and Yang, J. 2025. Visual Instruction Pretraining for Domain-Specific Foundation Models. arXiv. Lin, M. 2013. Network in network. arXiv. Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Doll´ar, P. 

2017. Focal Loss for Dense Object Detection. In ICCV. Lin, X.; Zhang, B.; Wu, F.; Wang, C.; Yang, Y.; and Chen, H. 

2023. SIVED: A SAR Image Dataset for Vehicle Detection Based on Rotatable Bounding Box. Remote Sensing. Liu, J.; Chen, H.; and Wang, Y. 2021. Multi-source remote sensing image fusion for ship target detection and recogni- tion. Remote Sensing, 13(23): 4852. Liu, J.-J.; Hou, Q.; Cheng, M.-M.; Wang, C.; and Feng, J. 2020. Improving Convolutional Networks With Self- Calibrated Convolutions. In CVPR. Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin, S.; and Guo, B. 2021. Swin transformer: Hierarchical vision transformer using shifted windows. In CVPR. Liu, Z.; Mao, H.; Wu, C.-Y.; Feichtenhofer, C.; Darrell, T.; and Xie, S. 2022. A convnet for the 2020s. In CVPR. Nakano, A.; Chen, S.; and Demachi, K. 2021. Cross- task consistency learning framework for multi-task learning. arXiv. Ni, K.; Zou, M.; Li, Y.; Li, X.; Guo, K.; Cheng, M.-M.; and Dai, Y. 2025. DenoDet V2: Phase-Amplitude Cross Denois- ing for SAR Object Detection. AAAI. Ren, S.; He, K.; Girshick, R.; and Sun, J. 2015. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In NeurIPS. Riquelme, C.; Puigcerver, J.; Mustafa, B.; Neumann, M.; Jenatton, R.; Susano Pinto, A.; Keysers, D.; and Houlsby, N. 2021. Scaling vision with sparse mixture of experts. NeurIPS. Sener, O.; and Koltun, V. 2018. Multi-task learning as multi- objective optimization. Advances in neural information pro- cessing systems. Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.; Hinton, G.; and Dean, J. 2017. Outrageously large neu- ral networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538. 


Sun, X.; Wang, P.; Yan, Z.; Xu, F.; Wang, R.; Diao, W.; Chen, J.; Li, J.; Feng, Y.; Xu, T.; Weinmann, M.; Hinz, S.; Wang, C.; and Fu, K. 2022a. FAIR1M: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery. ISPRS. 

Sun, Y.; Cao, B.; Zhu, P.; and Hu, Q. 2022b. Dronebased RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning. IEEE Transactions on Circuits and Systems for Video Technology, 1–1. 

Wang, C.; Ruan, R.; Zhao, Z.; Li, C.; and Tang, J. 2023. Category-oriented Localization Distillation for SAR Object Detection and A Unified Benchmark. IEEE Transactions on Geoscience and Remote Sensing. 

Wang, W.; Xie, E.; Li, X.; Fan, D.-P.; Song, K.; Liang, D.; Lu, T.; Luo, P.; and Shao, L. 2022. Pvt v2: Improved baselines with pyramid vision transformer. Computational Visual Media. 

Wang, X.; Cai, Z.; Gao, D.; and Vasconcelos, N. 2019. Towards universal object detection by domain attention. In CVPR. 

Wei, S.; Zeng, X.; Qu, Q.; Wang, M.; Su, H.; and Shi, J. 

2020. HRSID: A high-resolution SAR images dataset for ship detection and instance segmentation. IEEE Access. 

Xia, G.-S.; Bai, X.; Ding, J.; Zhu, Z.; Belongie, S.; Luo, J.; Datcu, M.; Pelillo, M.; and Zhang, L. 2018. DOTA: A Large-Scale Dataset for Object Detection in Aerial Images. In CVPR. 

Xia, R.; Chen, J.; Huang, Z.; Wan, H.; Wu, B.; Sun, L.; Yao, B.; Xiang, H.; and Xing, M. 2022. CRTransSar: A visual transformer based on contextual joint representation learning for SAR ship detection. Remote Sensing. 

Xie, X.; Cheng, G.; Wang, J.; Li, K.; Yao, X.; and Han, J. 

2024. Oriented r-cnn and beyond. IJCV. 

Xu, H.; Fang, L.; Liang, X.; Kang, W.; and Li, Z. 2020. Universal-rcnn: Universal object detector via transferable graph r-cnn. In AAAI. 

Yan, K.; Cai, J.; Zheng, Y.; Harrison, A. P.; Jin, D.; Tang, Y.; Tang, Y.; Huang, L.; Xiao, J.; and Lu, L. 2020. Learning from multiple datasets with heterogeneous and partial labels for universal lesion detection in CT. IEEE Transactions on Medical Imaging. 

Yang, G.; Manela, J.; Happold, M.; and Ramanan, D. 2019. Hierarchical deep stereo matching on high-resolution images. In CVPR. 

Yang, X.; Yan, J.; Ming, Q.; Wang, W.; Zhang, X.; and Tian, Q. 2021. Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss. In ICML. 

Yang, Y.; Jiang, P.-T.; Hou, Q.; Zhang, H.; Chen, J.; and Li, B. 2024. Multi-Task Dense Prediction via Mixture of LowRank Experts. In CVPR. 

Yuan, X.; Zheng, Z.; Li, Y.; Liu, X.; Liu, L.; Li, X.; Hou, Q.; and Cheng, M.-M. 2025. Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection. arXiv preprint arXiv:2501.03775. 

Zhang, H.; Huang, S.-L.; and Kuruoglu, E. E. 2024. HGR Correlation Pooling Fusion Framework for Recognition and 

 Classification in Multimodal Remote Sensing Data. Remote Sensing, 16(10): 1708. Zhang, P.; Xu, H.; Tian, T.; Gao, P.; Li, L.; Zhao, T.; Zhang, N.; and Tian, J. 2022. SEFEPNet: Scale expansion and feature enhancement pyramid network for SAR aircraft detection with small sample dataset. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. Zhang, T.; Zhang, X.; Li, J.; Xu, X.; Wang, B.; Zhan, X.; Xu, Y.; Ke, X.; Zeng, T.; Su, H.; et al. 2021. SAR ship detection dataset (SSDD): Official release and comprehensive data analysis. Remote Sensing. Zhang, X.; Li, D.; Dong, X.; Wu, T.; Yu, H.; Wang, J.; Li, Q.; and Li, X. 2025. UniChange: Unifying Change Detection with Multimodal Large Language Model. arXiv preprint arXiv:2511.02607. Zhang, Y.; and Yang, Q. 2021. A survey on multi-task learning. IEEE transactions on knowledge and data engineering. Zhang, Z.; Zhang, L.; Wu, J.; and Guo, W. 2024. Optical and Synthetic Aperture Radar Image Fusion for Ship Detection and Recognition: Current state, challenges, and future prospects. IEEE Geoscience and Remote Sensing Magazine. Zhao, X.; Schulter, S.; Sharma, G.; Tsai, Y.-H.; Chandraker, M.; and Wu, Y. 2020. Object detection with a unified label space from multiple datasets. In 2020. Zhou, X.; Koltun, V.; and Kr¨ahenb¨uhl, P. 2022. Simple multi-dataset detection. In CVPR. 


## A. Datasets 

The SARDet-100K (Li et al. 2024c) dataset is a SAR object detection dataset containing six object categories: Aircraft, Ship, Car, Bridge, Tank, and Harbor. The dataset consists of 94,493 training images with 198,747 instances, and 11,613 testing images with 24,023 instances. All annotations are provided as horizontal bounding boxes (HBB). DOTA (Xia et al. 2018) is an optical aerial object detection dataset that includes 15 categories. After splitting each image into 800×800 patches with a 400-pixel overlap, the dataset yields 25,028 training images containing 337,728 instances, and 17,041 testing images with 95,380 instances. All annotations are in the form of oriented bounding boxes (OBB). To avoid the severe dataset imbalance in the merged SOI-Det dataset, we use only a subset of SARDet-100K. For more details please refer to the Supplementary Material. DroneVehicle (Sun et al. 2022b) is an infrared vehicle detection dataset with 5 categories: car, truck, bus, van, and freight car. The images are sized at 640×512 pixels. The dataset consists of 17,990 training images with 316,411 instances, and 8,980 testing images with 159,616 instances. Similar to DOTA-v1.0, all annotations are in the form of OBB. To avoid the severe dataset imbalance in the merged SOI-Det dataset, we use only a subset of SARDet-100K: HRSID (Wei et al. 2020), MSAR (Xia et al. 2022), SADD (Zhang et al. 2022), OGSOD (Wang et al. 2023), and SIVED (Lin et al. 2023). This subset of SARDet-100K includes 47,097 training images with 125,462 annotated instances and 4,481 testing images with 12,566 instances. During training, each batch is sampled uniformly across the three datasets—SARDet-100K, DOTA, and DroneVehicle—in a 2:1:1 ratio, ensuring that each dataset is cycled through approximately once every 20K iterations. For each dataset, we evaluate using the mean Average Precision at IoU thresholds 0.5 (@50), at IoU thresholds 0.75 (@75), and at IoU thresholds from 0.5 to 0.95 (mAP). Additionally, we report the overall mAP across the three datasets to assess overall performance. 

## B. Implementation Details 

All models are fine-tuned on their respective training sets. For SARDet-100K and DroneVehicle, models are evaluated on the corresponding test sets, while for the DOTA dataset, evaluation is conducted on the validation set. For individual dataset training, models are trained for 12 epochs using the AdamW optimizer. In multi-modal joint training, we ensure that the total number of iterations matches the combined iterations of individual dataset training for fairness. Each batch employs uniform sampling from the three datasets (SARDet-100K, DOTA, and DroneVehicle) with a ratio of 2:1:1, ensuring that all datasets are cycled through approximately once every 20K iterations. Following previous network designs (Wang et al. 2019; Zhou, Koltun, and Kr¨ahenb¨uhl 2022), we share the backbone network and use separate task heads for different datasets and tasks. Specifically, after feature extraction by the backbone network, features from SARDet

 100K images are passed to the GFL head (due to its superior performance on horizontal SAR object detection (Li et al. 2024c)), while those from DOTA and DroneVehicle are passed to two individual O-RCNN (Xie et al. 2024) heads (due to the high-performance of O-RCNN on oriented object detection). In the main results and ablation studies, ConvNext-T is used as the default backbone unless otherwise specified. We run every experiment once. The initial learning rate is set to 0.0001, with a weight decay of 0.05. Model training is conducted using 8 RTX 3090 GPUs, with a batch size of 4 per GPU. All FLOPs reported in this paper are calculated using an 800×800 image input. 

## C. Grid-level sparse MoE analysis 

 MoE Layer Positions. We conduct an ablation study to assess the impact of incorporating MoE layers at different stages of the ConvNext backbone. As shown in Table 4, selectively adding MoE layers in the last three stages enhances model performance, resulting in a 1.02% mAP improvement with a minor increase in computational cost. This enhancement likely results from the richer semantic information in deeper stages, allowing experts to specialize more effectively. Conversely, abusing MoE layers does not lead to optimal performance, indicating that too many experts may introduce optimization challenges, as also highlighted in recent studies (Riquelme et al. 2021; Li et al. 2023a). Grid-level Expert Activation. Visualization of grid expert activation across the last three stages of a welltuned ConvNext-T backbone on DOTA, DroneVehicle and SARDet-100K images are given in Figure 7, 8 and 9. Each square grid represents the receptive field at a given stage, with different colours indicating the local grid areas processed by distinct experts. The top-1 selected experts for each grid are shown. In addition to the visualization, we further analyze the behaviour of sparse MoE expert selection at the dataset scale. We pass all test images from SARDet-100K, DOTA, and DroneVehicle datasets through the well-trained SM3Det model and gather statistics on expert participation. Specifically, the participation of an expert in a given MoE layer is quantified by the softmax probability of the gate function in main paper Eq. (2). The statistical results, presented in Figure 10, These findings validate the advantage of our sparse MoE design in addressing the M2Det task. The expert selection patterns demonstrate that some experts contribute to shared representation learning across all three modalities, while others specialize in distinct patterns, activating only specific modalities. This balance supports both joint and independent representation learning. Notably, experts activated by SAR images typically show low activation in the other two modalities, indicating that SAR images utilize a distinct set of experts. In contrast, Infrared and RGB images share several highly activated experts, reflecting greater overlap in their representations. This observation aligns with the conventional understanding that SAR imagery embodies unique concepts and characteristics distinct from those in other modalities. 


 Stage 1 Stage 2 Stage 3 Stage 4 FLOPs #P mAP @50 @75 None None None None 403G 66M 48.51 79.70 51.78 None None None Even 422G 132M 48.85+(0.34) 80.07(+0.37) 51.86+(0.08) None None Even Even 469G 174M 49.31+(0.80) 80.26+(0.56) 52.84+(1.06) None Even Even Even 487G 178M 49.53+(1.02) 80.47+(0.77) 53.06+(1.28) Even Even Even Even 506G 179M 49.47+(0.96) 80.33+(0.63) 52.98+(1.20) All All All All 572G 249M 49.30 +(0.79) 80.23+(0.53) 53.03+(1.25) 

Table 4: Experiments on spatial MoE with different MoE layer positions. “None”: no MoE layers, “Even”: MoE added to evenindexed layers, and “All”: MoE added to all layers within the stage. Each MoE layer comprises 8 experts with a top-2 selection. Selectively incorporating MoE layers in the even-indexed layers of the last three stages enhances model performance. 

 Input Image Stage 2 Experts Stage 3 Experts Stage 4 Experts Experts 

#### 1 ： 

#### 8 ： 

#### 2 ： 

#### 3 ： 

#### 4 ： 

#### 5 ： 

#### 6 ： 

#### 7 ： 

Figure 7: Visualization of grid-level expert activation across the last three stages of a well-tuned ConvNext-T backbone on DOTA-v1.0 images. Each square grid represents the receptive field at a given stage, with different colours indicating the local grid areas processed by distinct experts. The top-1 selected experts for each grid are shown. 

## D. Detailed Experiment Results 

### D.1 Parameter Efficiency. 

 Size(M) mAP 3 models SM3Det 3 models SM3Det 192 S 178 T 49.17 50.20 309 B 275 S 50.18 50.28 636 L 459 B 50.5 51.33 

- 770 L - 52.16 

Table 5: T, S, B, L: Backbone with ConvNext size of Tiny, Small, Base and Large. SM3Det consistently outperforms baseline models, even with fewer parameters. Its advantage increases with larger models. This demonstrates SM3Det’s robustness and effectiveness. 

 The results in Table 5 demonstrate SM3Det’s superior performance and parameter efficiency. Notably, SM3Det achieves performance gains even when compared to larger baseline models, underscoring the critical role of its consistency and synchronization optimization strategy rather than mere parameter scaling. For instance, SM3Det-Tiny (178M, T) outperforms the baseline Small model (192M, S) with a 1.03% mAP improvement (50.20 vs. 49.17) despite having 7.3% fewer parameters, highlighting that performance gains stem from effective multi-modal learning rather than parameter inflation. Similarly, SM3Det-Small (275M, S) surpasses the baseline Base model (309M, B) by 0.1% mAP (50.28 vs. 50.18) with 11% fewer parameters, further validating that SM3Det’s advantages are not driven by model scale but by its ability to harmonize multi-modal and multi-task optimization. 


 Input Image Stage 2 Experts Stage 3 Experts Stage 4 Experts Experts 

#### 1 ： 

#### 8 ： 

#### 2 ： 

#### 3 ： 

#### 4 ： 

#### 5 ： 

#### 6 ： 

#### 7 ： 

Figure 8: Visualization of grid-level expert activation across the last three stages of a well-tuned ConvNext-T backbone on DroneVehicle images. Each square grid represents the receptive field at a given stage, with different colours indicating the local grid areas processed by distinct experts. The top-1 selected experts for each grid are shown. 

The scalability advantage becomes more pronounced in larger configurations: SM3Det-Base (459M, B) achieves a 0.83% mAP gain (51.33 vs. 50.5) over the baseline Large model (636M, L), using 27.8% fewer parameters. Remarkably, SM3Det-Large (770M, L) can easily keep scaling up, it sets a new state-of-the-art (52.16 mAP) without a direct baseline counterpart, demonstrating its unique capacity to leverage cross-modal synergies. These results confirm that the grid-level sparse MoE backbone, coupled with dynamic learning rate synchronization, mitigates parameter conflicts and ensures balanced learning across modalities and tasks. Crucially, even in cross-scale comparisons (e.g., SM3Det-Small vs. baseline-Base), SM3Det maintains superior efficiency and accuracy, proving that its performance stems from systematic optimization rather than brute-force parameter scaling. This aligns with the core innovation of SM3Det: unifying multi-modal learning while preserving task-specific discriminability through adaptive optimization. 

### D.2 Ablation study on MoE configurations 

We investigate the impact of different configurations of the sparse MoE architecture on model performance. Specifically, we analyze variations in the number of experts and the Top-K value, which determines the number of experts activated for each input. The detailed results are given in Table 6. As the number of experts increases from 2 to 10 with a fixed Top-K value of 2, there is a consistent improvement in 

 overall mAP scores until reaching 8 experts. This suggests that up to a certain point, adding more experts allows the model to better capture diverse patterns in the data, leading to improved detection performance across multiple datasets. When the number of experts is increased to 10, the overall mAP slightly decreases to 49.24. This indicates that while adding more experts can enhance model capacity, there may be diminishing returns or even a negative impact when the number of experts exceeds the capacity of the training data to sufficiently train them all. When examining the impact of the Top-K value, it is clear that setting Top-K to 2 generally provides a good balance between performance and computational demand. For example, with 8 experts and a Top-K value of 2, the overall mAP is 49.31, which is higher than both the Top-K values of 1 and 3. A Top-K value of 1, while less computationally intensive, results in a slightly lower mAP of 49.05, indicating that activating only one expert per input may limit the model’s capacity to leverage the diverse expertise available. Conversely, increasing the Top-K value to 3, while introducing more computational demands, does not improve performance, as the overall mAP drops to 49.13. This suggests that activating only one expert per input may limit the model’s capacity to leverage the diverse expertise available, while activating too many experts for a single input may lead to over-complexity without corresponding benefits, potentially causing interference between experts or inefficient use of computational resources. 


 MoE Cfg FLOPs #P Test mAP @50 @75 Overall 48.51 79.70 51.78 SARDet-100K 56.99 87.32 61.41 DOTA 45.59 77.86 47.01 

 w/o MoE 403G 66M 

 DroneVehicle 47.09 76.10 54.53 Overall 48.94 80.25 52.01 SARDet-100K 57.81 88.33 61.96 DOTA 45.89 78.34 47.15 

 Experts: 2 Top: 2 469G^ 82M DroneVehicle 47.42 76.28 54.65 Overall 49.11 80.10 52.13 SARDet-100K 58.52 88.39 62.47 DOTA 45.84 77.99 46.98 

 Experts: 4 Top: 2 469G^ 113M DroneVehicle 47.64 76.49 55.17 Overall 49.11 79.74 52.76 SARDet-100K 59.16 88.89 64.10 DOTA 45.42 76.94 47.15 

 Experts: 6 Top: 2 469G^ 143M DroneVehicle 48.14 77.14 56.00 Overall 49.31 80.26 52.84 SARDet-100K 58.99 88.89 63.87 DOTA 45.69 77.64 47.13 

 Experts: 8 Top: 2 

#### 469G 174M 

 DroneVehicle 48.53 77.74 56.71 Overall 49.24 80.18 52.79 SARDet-100K 59.24 89.11 63.78 DOTA 45.65 77.54 47.27 

 Experts: 10 Top: 2 

#### 469G 205M 

 DroneVehicle 48.02 77.37 56.17 Overall 49.05 79.72 52.30 SARDet-100K 59.10 88.60 63.97 DOTA 45.23 76.88 46.31 

 Experts: 8 Top: 1 403G^ 174M DroneVehicle 48.44 77.60 56.28 Overall 49.31 80.26 52.84 SARDet-100K 58.99 88.89 63.87 DOTA 45.69 77.64 47.13 

 Experts: 8 Top: 2 469G^ 174M DroneVehicle 48.53 77.74 56.71 Overall 49.13 79.98 52.77 SARDet-100K 59.18 88.98 63.65 DOTA 45.43 77.32 47.33 

 Experts: 8 Top: 3 531G^ 174M DroneVehicle 48.16 77.15 56.05 Overall 48.49 79.60 51.51 SARDet-100K 56.88 87.17 60.49 DOTA 45.56 77.74 46.94 

 Image-level Experts: 3 Top: 1 

#### 403G 98M 

 DroneVehicle 47.21 76.07 54.45 Overall 48.60 79.67 52.06 SARDet-100K 56.51 87.23 61.00 DOTA 45.80 77.90 47.51 

 Grid level Experts: 3 Top: 1 

#### 403G 98M 

 DroneVehicle 47.50 75.93 54.96 

Table 6: Experiments on grid-level MoE with varying numbers of experts and top-K selection configurations. Experts are applied only to the even-indexed layers of the last two stages for validation efficiency. 

In summary, the ablation study reveals that the optimal configuration for this sparse MoE architecture in terms of balancing performance and computational efficiency is 8 experts with a Top-K value of 2. This configuration maximizes the model’s ability to learn from diverse inputs without introducing unnecessary complexity or overfitting. 

### D.3 Ablation on DSO Hyperparameters 

 We conduct an ablation study to evaluate the sensitivity of two key hyperparameters—temperature (τ ) and bias (b)—in the proposed DSO method, which dynamically adjusts learning rates to accommodate varying learning difficulties across tasks and modalities. The detailed results are given in Table 7. 


 T b Test mAP @50 @75 Overall 49.47 80.33 52.98 SARDet-100K 58.97 88.82 63.40 DOTA 46.02 77.84 47.73 

 w/o DSO 

 DroneVehicle 48.41 77.62 56.23 Overall 50.14 80.61 53.81 SARDet-100K 60.86 90.02 66.01 3 0.3 DOTA 46.23 77.68 47.81 DroneVehicle 49.03 77.99 56.90 Overall 50.20 80.68 53.79 SARDet-100K 60.64 89.94 65.06 3 0.4 DOTA 46.47 77.88 48.24 DroneVehicle 48.87 77.99 56.90 Overall 50.07 80.66 54.00 SARDet-100K 60.77 90.04 65.74 3 0.5 DOTA 46.11 77.70 48.16 DroneVehicle 49.11 78.28 57.43 Overall 50.03 80.61 53.98 SARDet-100K 60.49 89.92 65.30 DOTA 46.19 77.68 48.39 

#### 3 0.6 

 DroneVehicle 48.98 78.11 57.14 Overall 49.92 80.55 59.56 SARDet-100K 60.95 90.71 65.90 2 0.4 DOTA 45.85 77.35 47.81 DroneVehicle 48.89 77.98 87.05 Overall 50.20 80.68 53.79 SARDet-100K 60.64 89.94 65.06 3 0.4 DOTA 46.47 77.88 48.24 DroneVehicle 48.87 77.99 56.90 Overall 50.03 80.44 53.79 SARDet-100K 61.23 90.59 66.32 4 0.4 DOTA 45.86 77.13 47.58 DroneVehicle 49.08 78.19 57.39 

 Table 7: Experiments on the DSO method with varying temperature (τ ) and bias (b). 

The bias parameter b serves as a reweighting balance point. When the consistency score equals b, the reweighting factor is 1. Our analysis shows that a bias value of b = 0. 4 yields optimal performance, particularly when τ is set to 3, achieving an overall mean Average Precision (mAP) of 50.20%. This configuration also leads to the highest performance on the SARDet-100K dataset (60.64% mAP) and DroneVehicle (49.03% mAP). Variations in bias from 0.3 to 0.5 result in marginal performance changes, indicating robustness to bias adjustments. 

Regarding temperature, τ shapes the reweighting curve, affecting how sensitive learning rates adapt. A temperature of τ = 3 strikes an ideal balance between stability and responsiveness, as evidenced by the peak overall mAP. Lowering τ to 2 decreases the overall mAP to 49.92% and impairs task performance. Conversely, increasing τ to 4 results in a slight decline to 50.03%, suggesting that excessive smoothing or sharping hinders the model’s ability to react to varying task difficulties. 

In summary, the combination of τ = 3 and b = 0. 4 yields the best performance, effectively managing learning rate ad

 justments across diverse tasks and datasets. This study highlights the importance of tuning these hyperparameters to optimize the DSO method’s capabilities in multi-modal object detection scenarios. 

### D.4 Detailed results on different backbones 

 The performance of SM3Det across various backbones, including ConvNext, VAN, LSKNet, and PVT-v2, demonstrates its robustness and adaptability in different scenarios. As shown in Table 8, 9 and 10, SM3Det consistently outperforms individual models in terms of mean Average Precision (mAP) across all datasets. Specifically, SM3Det shows a notable improvement in overall mAP, particularly at higher IoU thresholds (@50 and @75), indicating its effectiveness in precise object detection tasks. Among the evaluated backbones, SM3Det paired with the ConvNext-B backbone achieved the highest overall mAP of 51.33% with a strong performance across all individual datasets, particularly SARDet-100K where it attained a mAP of 65.20%. This indicates that the backbone’s advanced feature extraction capabilities, when combined with 


SM3Det, significantly enhance detection accuracy. Similarly, SM3Det with VAN-B and LSKNet-B yield competitive results, with overall mAPs of 49.43% and 49.42%, respectively, further showcasing the model’s scalability and generalization capabilities. These results confirm that SM3Det not only scales effectively with larger backbone models but also generalizes well across different architectural designs, making it a versatile solution for various object detection tasks in remote sensing. 

### D.5 Detailed results on different detectors 

Based on the results presented in Table 12, the SM3Det framework consistently outperforms individual models across all evaluated detector combinations. The performance improvement is evident in both one-stage (GFL (Li et al. 2022), Retina (Lin et al. 2017), S^2 ANet (Han et al. 2020)) and two-stage (F-RCNN (Ren et al. 2015), Cascade FRCNN (Cai and Vasconcelos 2018), RoI-Trans (Ding et al. 2019), O-RCNN (Xie et al. 2024)) detectors, where SM3Det achieves higher mAP scores, particularly at higher IoU thresholds (@75). These results underscore the effectiveness of SM3Det’s architecture in leveraging the strengths of multiple detectors, leading to superior object detection performance across diverse datasets like SARDet-100K, DOTA, and DroneVehicle. The overall mAP improvements further validate the robustness of SM3Det in enhancing detection accuracy. 


 Backbone Model Test mAP @50 @75 Overall 48.23 79.39 51.26 SARDet-100K 57.31 87.44 61.99 DOTA 45.31 77.70 46.45 

 individual models DroneVehicle 46.09 74.78 52.79 Overall 50.24 80.68 53.81 SARDet-100K 60.64 89.94 65.06 DOTA 46.47 77.88 48.24 

 ConvNext-T 

 SM3Det DroneVehicle 48.87 77.99 56.90 Overall 49.17 80.06 52.31 SARDet-100K 60.62 89.54 64.85 DOTA 45.24 77.71 45.45 

 individual models DroneVehicle 47.22 75.72 54.82 Overall 50.28 80.13 54.30 SARDet-100K 62.98 91.49 68.77 DOTA 45.33 76.03 46.98 

 ConvNext-S 

 SM3Det 

 DroneVehicle 49.89 78.79 58.88 Overall 50.18 80.53 53.75 SARDet-100K 62.27 90.40 66.87 DOTA 46.02 77.79 47.73 

 individual models DroneVehicle 48.14 76.89 56.05 Overall 51.33 80.77 55.51 SARDet-100K 65.20 92.41 70.02 DOTA 45.86 76.35 48.10 

 ConvNext-B 

 SM3Det DroneVehicle 51.09 80.07 60.34 

 Table 8: Performance comparison of SM3Det with individual models across different ConvNext backbone scales. 

 Input Image Stage 2 Experts Stage 3 Experts Stage 4 Experts Experts 

#### 1 ： 

#### 8 ： 

#### 2 ： 

#### 3 ： 

#### 4 ： 

#### 5 ： 

#### 6 ： 

#### 7 ： 

Figure 9: Visualization of grid-level expert activation across the last three stages of a well-tuned ConvNext-T backbone on SARDet-100K images. Each square grid represents the receptive field at a given stage, with different colours indicating the local grid areas processed by distinct experts. The top-1 selected experts for each grid are shown. 


Figure 10: Expert participation statistics across SARDet-100K, DOTA, and DroneVehicle datasets. Some experts contribute to shared representation learning across all modalities, while others specialize in distinct patterns, activating only specific modalities. 

 Backbone Model Test mAP @50 @75 Overall 44.15 75.21 46.77 SARDet-100K 46.74 77.63 49.71 DOTA 43.31 75.36 44.73 

 individual models DroneVehicle 43.55 71.86 49.38 Overall 45.46 76.28 48.37 SARDet-100K 49.28 80.85 52.08 DOTA 43.60 74.73 45.22 

#### VAN-T 

 SM3Det DroneVehicle 46.47 75.43 53.37 Overall 47.47 78.78 50.82 SARDet-100K 52.71 83.82 57.74 DOTA 45.47 77.67 46.91 

 individual models DroneVehicle 74.17 76.07 54.27 Overall 49.03 79.60 52.76 SARDet-100K 57.98 88.36 62.46 DOTA 45.50 76.66 47.37 

#### VAN-S 

 SM3Det 

 DroneVehicle 48.87 77.91 57.31 Overall 48.91 79.99 52.83 SARDet-100K 53.73 84.89 58.11 DOTA 47.42 79.24 49.86 

 individual models DroneVehicle 47.58 76.38 55.42 Overall 49.43 80.57 53.34 SARDet-100K 56.82 87.82 61.15 DOTA 46.62 78.58 48.92 

#### VAN-B 

 SM3Det DroneVehicle 48.98 77.82 57.24 

 Table 9: Performance comparison of SM3Det with individual models across different VAN backbone scales. 


 Backbone Model Test mAP @50 @75 Overall 44.71 75.86 47.31 SARDet-100K 47.24 78.24 50.39 DOTA 43.93 75.97 45.25 

 individual models DroneVehicle 44.00 72.67 49.78 Overall 45.64 76.89 48.25 SARDet-100K 49.95 81.76 53.61 DOTA 43.56 75.44 44.17 

 LSKNet-T 

 SM3Det DroneVehicle 46.71 75.42 54.04 Overall 47.49 78.65 51.22 SARDet-100K 53.12 84.33 57.64 DOTA 45.46 77.36 47.73 

 individual models DroneVehicle 46.80 75.72 54.01 Overall 48.79 79.78 52.42 SARDet-100K 58.41 88.48 62.83 DOTA 44.80 76.69 46.61 

 LSKNet-S 

 SM3Det DroneVehicle 49.20 78.63 57.36 Overall 48.58 79.43 51.89 SARDet-100K 54.00 84.81 58.46 DOTA 46.87 78.45 48.76 

 individual models DroneVehicle 47.19 75.91 51.35 Overall 49.42 80.34 53.31 SARDet-100K 56.70 87.75 60.93 DOTA 46.61 78.11 48.94 

 LSKNet-B 

 SM3Det 

 DroneVehicle 49.13 78.14 57.29 

Table 10: Performance comparison of SM3Det with individual models across different LSKNet backbone scales. 

 Backbone Model Test mAP @50 @75 Overall 43.46 75.33 45.39 SARDet-100K 47.63 78.83 50.96 DOTA 41.81 74.79 42.18 

 individual models DroneVehicle 43.39 72.73 48.36 Overall 44.58 76.43 46.81 SARDet-100K 48.58 80.71 52.31 DOTA 42.72 75.39 42.92 

 PVT-v2 T 

 SM3Det DroneVehicle 45.38 74.43 51.86 Overall 46.36 78.38 49.61 SARDet-100K 52.01 84.24 57.19 DOTA 44.28 77.14 45.60 

 individual models DroneVehicle 45.84 75.08 52.54 Overall 47.31 79.21 50.08 SARDet-100K 54.53 85.48 58.94 DOTA 44.37 77.53 45.01 

 PVT-v2 S 

 SM3Det DroneVehicle 47.47 76.71 54.66 Overall 48.51 80.23 52.05 SARDet-100K 56.04 87.00 61.27 DOTA 45.55 78.41 46.92 

 individual models DroneVehicle 48.37 77.54 56.39 Overall 49.34 80.72 53.05 SARDet-100K 57.34 87.75 62.24 DOTA 46.24 78.81 47.92 

 PVT-v2 B 

 SM3Det 

 DroneVehicle 49.06 78.02 57.41 

 Table 11: Performance comparison of SM3Det with individual models across different PVT-v2 backbone scales. 


 Detectors Model Test mAP @50 @75 Overall 48.23 79.39 51.26 SARDet-100K 57.31 87.44 61.99 DOTA 45.31 77.70 46.45 

 individual models DroneVehicle 46.09 74.78 52.79 Overall 50.20 80.68 53.81 SARDet-100K 60.64 89.94 65.06 DOTA 46.47 77.88 48.24 

 SARDet-100K: GFL DOTA: O-RCNN DroneVehicle: O-RCNN SM3Det DroneVehicle 48.87 77.99 56.90 Overall 45.67 77.92 47.22 SARDet-100K 51.08 82.50 55.27 DOTA 43.93 77.37 43.10 

 individual models DroneVehicle 44.42 74.06 49.94 Overall 48.01 78.83 51.35 SARDet-100K 53.04 83.99 57.76 DOTA 45.43 76.79 46.50 

 SARDet-100K: Retina DOTA: RoI-Trans DroneVehicle: RoI-Trans SM3Det DroneVehicle 49.69 78.76 58.23 Overall 47.10 78.61 50.17 SARDet-100K 52.40 84.09 57.28 DOTA 45.31 77.70 46.45 

 individual models DroneVehicle 46.09 74.78 52.79 Overall 48.89 79.39 53.20 SARDet-100K 54.56 85.62 59.83 DOTA 46.09 76.80 48.38 

 SARDet-100K: F-RCNN DOTA: O-RCNN DroneVehicle: O-RCNN SM3Det DroneVehicle 50.50 79.69 59.68 Overall 46.50 78.32 48.24 SARDet-100K 54.65 84.26 59.66 DOTA 43.93 77.37 43.10 

 individual models DroneVehicle 44.42 74.06 49.94 Overall 48.59 79.04 51.69 SARDet-100K 56.30 85.39 61.46 DOTA 45.00 76.46 45.23 

 SARDet-100K: Cascade DOTA: RoI-Trans DroneVehicle: RoI-Trans SM3Det DroneVehicle 50.11 79.16 59.32 Overall 42.29 77.18 40.68 SARDet-100K 52.40 84.09 57.28 DOTA 38.47 75.81 32.82 

 individual models DroneVehicle 41.64 72.98 44.35 Overall 43.12 77.40 42.83 SARDet-100K 49.20 81.68 53.13 DOTA 39.92 76.20 35.87 

 SARDet-100K: F-RCNN DOTA: S^2 ANet DroneVehicle: S^2 ANet SM3Det DroneVehicle 45.44 75.85 51.33 Overall 42.81 77.22 41.23 SARDet-100K 54.65 84.26 59.66 DOTA 38.47 75.81 32.82 

 individual models DroneVehicle 41.64 72.98 44.35 Overall 43.76 77.28 43.63 SARDet-100K 53.11 82.73 57.98 DOTA 39.51 75.61 35.32 

 SARDet-100K: Cascade DOTA: S^2 ANet DroneVehicle: S^2 ANet SM3Det DroneVehicle 45.27 75.76 51.32 Overall 43.43 77.95 41.77 SARDet-100K 57.31 87.44 61.99 DOTA 38.47 75.81 32.82 

 individual models DroneVehicle 41.64 72.98 44.35 Overall 45.25 78.97 45.07 SARDet-100K 59.01 88.77 63.84 DOTA 39.79 76.07 35.71 

 SARDet-100K: GFL DOTA: S^2 ANet DroneVehicle: S^2 ANet SM3Det DroneVehicle 45.13 75.92 50.65 Overall 41.99 76.81 40.22 SARDet-100K 51.08 82.50 55.27 DOTA 38.47 75.81 32.82 

 individual models DroneVehicle 41.64 72.98 44.35 Overall 43.13 77.62 42.67 SARDet-100K 50.63 82.04 54.82 DOTA 39.45 76.44 35.15 

 SARDet-100K: Retina DOTA: S^2 ANet DroneVehicle: S^2 ANet SM3Det DroneVehicle 45.15 75.83 50.67 

Table 12: Performance comparison of SM3Det with individual models across various detector combinations. 


