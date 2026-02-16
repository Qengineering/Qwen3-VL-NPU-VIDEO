# Qwen3-VL NPU VIDEO

![Sequence](https://github.com/user-attachments/assets/1e6af4e2-cdc4-4e22-868a-82e3df34fbb7)<br><br>
User: <image> Describe the video.<br><br>
Answer: The video shows a man performing a martial arts move in a large, open room with a light blue wall and green mat flooring.

- The man is wearing dark clothing.
- He begins by standing on his right foot, then performs a high kick to his left leg while simultaneously bending forward at the waist.
- As he executes this movement, he appears to be performing a martial arts technique that involves a combination of kicks and body control.
- After completing the move, he falls backwards onto the mat, landing on his back with his legs extended.

The video captures the man's dynamic motion as he performs the martial arts move.

------------

## Qwen3 VLM VIDEO for RK3588 NPU (Rock 5, Orange Pi 5). <br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/><br/>
Paper: [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)<br/><br/>
Hugging face: https://huggingface.co/collections/Qwen/qwen3-vl

------------

## Introduction

LLMs (Large Language Models) are neural networks trained on large text datasets to understand and generate language.<br>
VLMs (Vision-Language Models) add a visual encoder so the model can process images and text together.<br> 
A combined VLM+LLM system is often referred to as a multimodal model.

These models can be large—hundreds of millions to billions of parameters—which impacts accuracy, memory use, and runtime speed.<br>
On edge devices like the RK3588, available RAM and compute are limited, and even the NPU has strict constraints on supported operations.<br>
Because of this, models typically need to be quantised or simplified to fit.

Performance is usually expressed in tokens (words) per second.<br>
Once converted to RKNN, parts of the model can run on the NPU, improving speed.<br>

<img width="1216" height="737" alt="Qwen3_chart" src="https://github.com/user-attachments/assets/ed7ec7c9-39d9-49c1-b2e6-643c87583873" /><br><br>
#### ❗Showstopper❗
To process video input, individual frames are first extracted. The VLM converts each frame into embeddings, which are then transformed into vision tokens — as illustrated in the schema above.<br>
Even on a desktop PC, this process places a heavy load on memory and CUDA resources. It’s therefore no surprise that the Rock 5C, with its more limited hardware, struggles even more.<br>
The RKLLM library supports a maximum of 4092 tokens in total. Each vision token corresponds to about 200 tokens, limiting processing to **20** frames per video. To stay within this constraint, the video is subsampled, and evenly spaced frames are extracted for processing by Qwen3.<br><br>
For reference, each vision token occupies around 20 MB of RAM — a detail worth keeping in mind when working on systems with limited memory.

------------

## Model performance benchmark (FPS)

All models below can handle multiple frames. Best performing is **Qwen3**, as the 'parent' of InternVL3.5 and SmolVLM2.<br><br>
All LLM models are quantized to **w8a8**, while the VLM vision encoders use **fp16**.<br>

| model         | RAM (GB)<sup>1</sup> | llm cold sec<sup>2</sup> | llm warm sec<sup>3</sup> | vlm cold sec<sup>2</sup> | vlm warm sec<sup>3</sup> | Resolution | Tokens/s |
| --------------| :--: | :-----: | :-----: | :--------: | :-----: | :--------:  | :--------: |
| [Qwen3-2B](https://github.com/Qengineering/Qwen3-VL-2B-NPU) | 3.1 | 21.9 | 2.6 | 10.0  | 0.9 | 448 x 448 | 11.5 |
| [Qwen3-4B](https://github.com/Qengineering/Qwen3-VL-4B-NPU) | 8.7 | 49.6 | 5.6 | 10.6  | 1.1 | 448 x 448 | 5.7 |
| [InternVL3.5-1B](https://github.com/Qengineering/InternVL3.5-1B-NPU) | 1.9 |  8.3 |   8.0 | 1.5    | 0.8 | 448 x 448 | 24 |
| [InternVL3.5-2B](https://github.com/Qengineering/InternVL3.5-2B-NPU) | 3.0 |  22 |   8.0 | 2.7    | 0.8 | 448 x 448 | 11.2 |
| [InternVL3.5-4B](https://github.com/Qengineering/InternVL3.5-4B-NPU) | 5.4 |  50 |   8.0 | 5.9    | 0.8 | 448 x 448 | 5 |
| [InternVL3.5-8B](https://github.com/Qengineering/InternVL3.5-8B-NPU) | 8.8 |  92 |   8.0 | 50.5    | 5.8 | 448 x 448 | 3.5 |
| [SmolVLM2-2.2B](https://github.com/Qengineering/SmolVLM2-2B-NPU) | 3.4 | 21.2 |   2.6 | 10.5   | 0.9  | 384 x 384 | 11 |
| [SmolVLM2-500M](https://github.com/Qengineering/SmolVLM2-500M-NPU) | 0.8 |  4.8 |   0.7 | 2.5    | 0.25 | 384 x 384 | 31 |
| [SmolVLM2-256M](https://github.com/Qengineering/SmolVLM2-256M-NPU) | 0.5 |  1.1 |   0.4 | 2.5    | 0.25 | 384 x 384 | 54 |

<sup>1</sup> The total used memory; LLM plus the VLM. <br>
<sup>2</sup> When an llm/vlm model is loaded for the first time from your disk to RAM or NPU, it is called a cold start.<br>
The duration depends on your OS, I/O transfer rate, and memory mapping.<br> 
<sup>3</sup> Subsequent loading (warm start) takes advantage of the already mapped data in RAM. Mostly, only a few pointers need to be restored.<br><br>
<img width="600" height="450" alt="Plot_1" src="https://github.com/user-attachments/assets/2dde8d27-c8ae-474c-b845-4ed52bdc0785" /><br>
<img width="600" height="450" alt="Plot_2" src="https://github.com/user-attachments/assets/0cf946d5-5458-4166-bc2b-fa1592ae4d6b" />

------------

## Dependencies.
To run the application, you have to:
- OpenCV 64-bit installed.
- rkllm library.
- rknn library.
- Optional: Code::Blocks. (```$ sudo apt-get install codeblocks```)

### Installing the dependencies.
Start with the usual 
```
$ sudo apt-get update 
$ sudo apt-get upgrade
$ sudo apt-get install cmake wget curl
```
#### OpenCV
To install OpenCV on your SBC, follow the Raspberry Pi 4 [guide](https://qengineering.eu/install-opencv-on-raspberry-64-os.html).<br><br>
Or, when you have no intentions to program code:
```
$ sudo apt-get install libopencv-dev 
```
------------

## Installing the app.
```
$ git clone https://github.com/Qengineering/Qwen3-VL-NPU-VIDEO
```

#### RKLLM, RKNN
To run InternVL3, you need to have the **rkllm-runtime** library version **1.2.3** (or higher) installed, as well as the **rknpu driver** version **0.9.8**.<br>
If you don't have these on your machine, or if you have a lower version, you need to install them.<br>
We have provided the correct versions in the repo.<br>
```bash
$ cd ./Qwen3-VL-2B-NPU/aarch64/library
$ sudo cp ./*.so /usr/local/lib
$ cd ../include
$ sudo cp ./*.h /usr/local/include
```
### Download the LLM and VLM model.
The next step is downloading the models.<br>
All models are on our Sync.com server. Please look at the repository of your choice for the appropriate links.<br>
For instance, Qwen3-VL-2B:<br>
[qwen3-vl-2b-instruct_w8a8_rk3588.rkllm](https://ln5.sync.com/dl/6cd2e45d0#swbgmrgn-xqjwb4pn-h3fizzg5-vb3jvfxd) and [qwen3-vl-2b-vision_rk3588.rknn](https://ln5.sync.com/dl/d1a22a380#kshmvhzf-ma8xhheb-mbx2x47f-qp5fajj5)<br>
Copy both into this folder.

### Building the app.
Once you have the two models, it is time to build your application.<br>
You can use **Code::Blocks**.
- Load the project file *.cbp in Code::Blocks.
- Select _Release_, not Debug.
- Compile and run with F9.
- You can alter command line arguments with _Project -> Set programs arguments..._ 

Or use **Cmake**.
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```
### Running the app.
The app supports two modes:

#### Video.

```bash
./VLM_VIDEO_NPU RKNN_model RKLLM_model Video Frames
```
| Argument   | Comment |
| --------------| --  |
| RKNN_model | The visual encoder model (vlm) | 
| RKLLM_model | The large language model (llm) | 
| Video | The video. Can be mp4, avi, mov, or mkv | 
| Frames | Optional, default 8| 

Each frame requires about 2.24 seconds of VLM loading time.<br>
Increasing the number of frames also extends the overall thinking time, since all corresponding vision tokens must be processed.

#### Files.

```bash
./VLM_VIDEO_NPU RKNN_model RKLLM_model File1 File2 File3 ... FileX
```
| Argument   | Comment |
| --------------| --  |
| RKNN_model | The visual encoder model (vlm) | 
| RKLLM_model | The large language model (llm) | 
| File | The individual image file. | 

Each file requires about 2.24 seconds of VLM loading time.<br>
Increasing the number of frames also extends the overall thinking time, since all corresponding vision tokens must be processed.

### Using the app.
Using the application is simple. Once you provide the video and the models, you can ask everything you want.<br>
Remember, we are on a bare Rock5C, so don't expect the same quality answers as ChatGPT can provide.<br>
If you want to talk about the video, you need to include the token `<image>` in your prompt once.<br>
The app remembers the dialogue until you give the token `<clear>`.<br>
With `<exit>`, you leave the application.

------------

## **[Rock5GPT](https://rock5gpt.qengineering.eu)**
To get a taste, try our professional Qwen3 AI-chatbot running on a Rock 5C: https://rock5gpt.qengineering.eu
<img width="815" height="1151" alt="Rock5GPT" src="https://github.com/user-attachments/assets/3ce5ad31-bc2b-4513-8ac9-42be793a86db" /><br>

------------

[![paypal](https://qengineering.eu/images/TipJarSmall4.png)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CPZTM5BB3FCYL) 
