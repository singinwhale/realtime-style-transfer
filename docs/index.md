---
title: Home
---

## Abstract

> In this work we explore the possibilities of using convolutional neural networks for style transfer in the context of
> a real-time deferred renderer like Unreal Engine 5. We explore the possibilities of using G-buffer data as input to the
> neural network to improve its capabilities over just the final RGB image. An initial implementation in Unreal Engine
> yields 50 frames per second. Incorporating the G-buffer data improves the output image quality of the network
> significantly.

[Thesis](https://github.com/singinwhale/realtime-style-transfer/releases/download/1.0.0/realtime-style-transfer.pdf) | [Code1](https://github.com/singinwhale/realtime-style-transfer) | [Code2](https://github.com/singinwhale/realtime-style-transfer-unreal) | [Code3](https://github.com/singinwhale/UnrealEngine/tree/realtime-style-transfer)

<iframe width="480" height="240"
src="https://www.youtube-nocookie.com/embed/Y437ejhyT_U" frameborder="0" allow="autoplay; encrypted-media"
allowfullscreen></iframe>

## Pretrained Models

Pretrained models are available in the releases section:

[Link](https://github.com/singinwhale/realtime-style-transfer/releases/tag/1.0.0)

## Code

### Network Training

The source code for getting the dataset and training the different variants of the network
can be found here:

[Link](https://github.com/singinwhale/realtime-style-transfer).

### Unreal Project


#### Unreal Engine Modifications

You need to be a member of the _Epic Games_ organization.
To get access follow these instructions: [Link](https://www.unrealengine.com/en-US/ue-on-github)

The engine modifications can be found in the UnrealEngine fork on the realtime-style-transfer branch.

[Link](https://github.com/singinwhale/UnrealEngine/tree/realtime-style-transfer)


The LFS assets for the Lyra project are too big for my GitHub so only parts of it are available on GitHub.

#### Plugin

The standalone style transfer **plugin** alone is
on my GitHub. 

[Link](https://github.com/singinwhale/realtime-style-transfer-unreal)

#### Full Lyra Project

The game project is available on my private GitTea instance.
You are still going to need the modified Unreal Engine to run this project. 

[Link](https://git.singinwhale.com/singinwhale/RealtimeStyleTransferRuntime)

## Videos

### rst-960-120-32-3 In Engine

<iframe width="480" height="240"
src="https://www.youtube-nocookie.com/embed/Y437ejhyT_U" frameborder="0" allow="autoplay; encrypted-media"
allowfullscreen></iframe>

### rst-960-120-128-18

<iframe width="480" height="240"
src="https://www.youtube-nocookie.com/embed/x51uoaF6rGY" frameborder="0" allow="autoplay; encrypted-media"
allowfullscreen></iframe>

### rst-960-120-32-18

<iframe width="480" height="240"
src="https://www.youtube-nocookie.com/embed/bjWtSlDXMBM" frameborder="0" allow="autoplay; encrypted-media"
allowfullscreen></iframe>

### rst-960-120-32-3

<iframe width="480" height="240"
src="https://www.youtube-nocookie.com/embed/bsCiJekEjrw" frameborder="0" allow="autoplay; encrypted-media"
allowfullscreen></iframe>

### rst-960-120-128-17

<iframe width="480" height="240"
src="https://www.youtube-nocookie.com/embed/upCAFem6tdE" frameborder="0" allow="autoplay; encrypted-media"
allowfullscreen></iframe>

### rst-960-120-32-17

<iframe width="480" height="240"
src="https://www.youtube-nocookie.com/embed/2mwpBTuS9M4" frameborder="0" allow="autoplay; encrypted-media"
allowfullscreen></iframe>

