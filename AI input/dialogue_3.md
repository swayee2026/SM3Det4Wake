# PLAN


## 任务介绍

我是一名研究生，我的课题方向是：遥感图像领域的目标识别。现在我准备完成我的第一批论文的写作，我的任务是：光学遥感图像中的船只尾迹目标检测。

我使用的基座模型是 SM3Det，我使用的数据集是开源的swimshipwake-imagery-mass。

SM3Det 这一篇论文的核心点在于，将 遥感 领域的3类常见模态串联了起来，也就是 optical infrared and SAR，他使用了多专家模块 MoE和一个router，多任务(HBB, OBB)的下游检测头，和一个Dynamic Submodule Optimization (DSO)可以动态差异化的调节每个检测头的学习率。

我的任务是：光学图像中的船只尾迹目标检测，因此是单模态，2种指定的目标类型。由于数据集的构造，head 部分依然是双任务。

基于此，我需要使用 sm3det 模型中的 grid level MoE 模块，包括 router and expert，用于学习到不同目标类型的特征 ( land, ocean, wake, ship, etc)；差异化的学习率模块 DSO （船只仅有点的标注，因此学习速度快于尾迹的bbox）;ConvNext 的框架 backbone, neck ,head 等大框架都不改变。


## SWIM-Ship Wake Imagery Mass 的介绍：
```txt
PASCAL VOC format
SWIM dataset is a benchmark dataset built for deep learning-based ship wake detection. It contains images of coastal areas around Asia, Europe, Africa, North America, and Oceania. All the images were taken from 2009 to 2021. The objects to be detected included the wakes of ships, ranging from small yacht to huge container ships, and the backgrounds included open seas, ports, straits, and canals. The images in the dataset have spatial resolutions of 2.5 m to 0.5 m. The image size is 768 × 768 pixels.
SWIM dataset contains 11,600 positive images and 3,010 negative images, and can provide up to 15,356 carefully annotated wake instances.
SWIM dataset uses an oriented bounding box and a set of landmarks to accurately locate the wake instance.
```

annotation sample （这是尾迹目标的标注信息）


```txt
<annotation>
  <folder>Positive</folder>
  <filename>00001</filename>
  <format>jpg</format>
  <source>
    <database>SWIM</database>
  </source>
  <size>
    <width>768</width>
    <height>768</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <type>robndbox</type>
    <name>wake</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <robndbox>
      <cx>602.8032</cx>
      <cy>53.0397</cy>
      <w>44.4618</w>
      <h>96.8959</h>
      <angle>0.53</angle>
    </robndbox>
  </object>
</annotation>

```

landmark sample （这是船只目标的标注信息）

```txt

<annotation>
	<folder>Positive</folder>
	<filename>00001</filename>
	<format>jpg</format>
	<source>
		<database>SWIM</database>
	</source>
	<size>
		<width>768</width>
		<height>768</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<type>pointtheta</type>
		<name>wake</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<pointtheta>
			<px>581.6883116883117</px>
			<py>83.01298701298701</py>
			<theta1>-1.2298173732985473</theta1>
			<theta2>-0.7488630676110335</theta2>
		</pointtheta>
	</object>
</annotation>
```


## 我需要的修改

1. residual block - addition

在 backbone 中的每一层，新增了残差连接 residual block，和 grid level MoE 模块并行，将全局信息引入到下一层的输入中。考虑到船只是稠密特征，而尾迹是稀疏特征，每一次下采样，都会先提取船只信息，然后是尾迹。因此我的残差连接上的变换（条带卷积，低频滤波）是交错设置的。我的backbone 共计4层。

| Stage | 下采样倍率 | 分辨率(800×800输入) | 变换类型 | 设计目标 |
|-------|-----------|-------------------|---------|---------|
| 1 | 4× | 200×200 | LowFreqResidual (5×5) | 保留船只高频纹理，平滑背景噪声 |
| 2 | 8× | 100×100 | StripConvResidual (1×7, 7×1) | 初步提取尾迹线性结构 |
| 3 | 16× | 50×50 | LowFreqResidual (7×7) | 强化深层船只语义纹理 |
| 4 | 32× | 25×25 | StripConvResidual (1×11, 11×1) | 强化大尺寸尾迹全局结构 |

该模块为我的论文的核心创新点之一

2. dual stream induction of attention mask -addition

船只目标和尾迹目标总是正对出现的，且二者存在很明确的几何位置的关系。我希望在backbone部分，每一层输出船只预测目标后，用于下一层的尾迹目标的注意力蒙版引导。尾迹同理，每一层输出尾迹预测目标后，用于下一层的船只目标的注意力蒙版引导。

每个目标类型生成3个通道
Mask_ship = [Confidence_map, Direction_map, Distance_transform]
Mask_wake = [Confidence_map, Direction_map, Distance_transform]
通道	维度	说明
Confidence_map	(H, W)	目标存在概率 [0,1]
Direction_map	(H, W, 2)	方向向量 (cosθ, sinθ)，表示航向
Distance_transform	(H, W)	到目标中心的距离场，编码空间关系


下一层融合公式改进：

原设计（仅位置）
F_wake_guided = F_input * (1 + α * Mask_ship_conf)

新设计（位置+方向）
directional_attention = Mask_ship_conf * (1 + cos(θ_ship - θ_wake))
F_wake_guided = F_input * (1 + α * directional_attention)

该模块为我的论文的核心创新点之一


3. detection head - modification

由于我的检测目标的数据格式不相同，因此，我需要修改 head 模块。其中一个按照正常的OBB检测模块实现，另一个则改为 3 元组的预测任务。二者对齐 sm3det 中已经实现的 差异化的学习率模块 DSO，实现异步学习率的调整 。
   1. 一个任务是尾迹目标的旋转目标框的学习（oriented bounding box detection, OBB）regression ，包含正常的5维数据(x,y,h,w,theta)
   2. 一个任务是尾迹目标的是否存在 class regression,包含1维数据(p)
   3. 一个是船只的点位置的学习 position regression，包含3维数据(x,y,theta)，其中角度 theta 只的是船只的方向，也就是尾迹的方向。

该模块为我的论文的核心创新点之一

4. visualization - addtion

我希望在论文中插入模型每一层的输出结果、注意力蒙版、MoE 的grid level 调用的结果。请你搜索现有代码，找出有无相关函数或者接口，已经实现可以直接调用的，如果有，请罗列出相关函数，并指出如何输出可视化的方案。如果没有，请你增加相关接口，用于输出 backbone 每一层的检测结果中间变量的效果图。


该工作不作为我的论文核心创新点，是实验的适配脚本


5. pipeline validation - addition

我希望设置一个脚本，用于验证我的数据通路的正确性。我会使用一个数据容量非常小的数据集 （10 张图片），仅用于验证数据通路和可视化相关接口是否正常输出，不涉及模型的训练效果。在该训练配置文件中，所有的参数量都设置为最小值,epoch =1, 等等。


该工作不作为我的论文核心创新点，是实验的适配脚本




