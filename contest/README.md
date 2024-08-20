

# Jittor 风格迁移图片生成赛题

![image](assest/image.png)

## 简介

本项目包含了第四届计图挑战赛风格迁移图片生成赛题的代码实现。本项目采用了多种方法对图片处理，取得了符合风格的效果。

## 安装 

本项目在3090上运行，训练一个风格从4分钟到1小时不等。

#### 运行环境

- ubuntu 20.04.6 LTS
- python == 3.9
- jittor == 1.3.10
- CUDA == 12.2

#### 安装依赖

安装官方baseline指导安装所有库，然后用libs中的文件替换你环境中相应的文件。

#### 预训练模型

我没有云盘空间，但是我传给了比赛官方，你可以联系他们以获得我提交的权重，将下载的weights文件夹放置在项目根目录下即可。

## 数据预处理

生成推理时固定的随机种子起始latents

```
bash scripts/gen.sh
```

## 训练

```
python train.py
```

## 推理

```
python test.py
```

## 致谢

JDiffusion ...... (to be update)
